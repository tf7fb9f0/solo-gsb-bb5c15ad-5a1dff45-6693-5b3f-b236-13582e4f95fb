"""会议室批量分配核心逻辑：会议室 upsert/查询与整批全局求解分配。

- 会议室：唯一 ``room_id``、容量与**去重排序**后的设备标签；
  upsert 幂等（不存在则创建，存在则整体替换容量与设备标签），
  查询支持按最小容量与必备设备过滤。
- 批量分配：提交一组 **active** 预约及各自所需容量/设备，
  在单个 ``BEGIN IMMEDIATE`` 事务内结合**已有分配**（既有批次落库的
  房间占用）与预约时段做**全局求解**：同一房间不得分给时间重叠
  （半开区间，首尾相接不算重叠）的预约。**仅当整批都有解时才持久化**；
  无解则整体回滚、已有分配保持不变，并以
  :class:`BatchUnsolvableError`（API 409）列出无法分配的预约。
- 结果**确定性**：完备解中先取**总空余座位最少**
  （Σ 房间容量 − 所需容量），再按（预约 id 升序、room_id 字典序）
  取字典序最小的解。无解时报告“可分配数最多、并列时总空余最少、
  再按同一向量字典序最小（未分配排在任何 room_id 之后）”的解里
  未获分配的预约——同一输入永远得到同一答案。
- 同一 ``batch_id`` 幂等：规范化内容（按预约 id 排序的
  （预约、所需容量、去重设备）列表）相同 → 重放返回该批次当前分配；
  内容不同 → :class:`BatchContentConflictError`（409）。
  重新提交包含某预约的新批次会在同事务内**替换**其旧分配；
  批次失败则旧分配原样保留。
- 预约**改期/取消/时段交换**会在各自事务内释放其会议室分配
  （分配与求解时的时段绑定，时段变更即失效），
  因此已有分配之间永远不会出现时间重叠。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from . import db
from .reservations import (
    ReservationCanceledError,
    ReservationNotFoundError,
    _canon,
    _parse,
)

_ROOM_COLUMNS = "room_id, capacity, equipment_key, created_at, updated_at"


class RoomNotFoundError(Exception):
    """指定 room_id 的会议室不存在。"""

    def __init__(self, room_id: str) -> None:
        self.room_id = room_id
        super().__init__(f"meeting room {room_id!r} not found")


class BatchNotFoundError(Exception):
    """指定 batch_id 的分配批次不存在。"""

    def __init__(self, batch_id: str) -> None:
        self.batch_id = batch_id
        super().__init__(f"room assignment batch {batch_id!r} not found")


class BatchContentConflictError(Exception):
    """``batch_id`` 已被内容不同的批量请求占用。"""

    def __init__(self, batch_id: str) -> None:
        self.batch_id = batch_id
        super().__init__(
            f"batch_id is already used by room assignment batch "
            f"{batch_id!r} with different content"
        )


class BatchUnsolvableError(Exception):
    """整批无完备解：携带确定性最优部分解中未获分配的预约 id（升序）。"""

    def __init__(self, unassignable: list[int]) -> None:
        self.unassignable = unassignable
        pretty = ", ".join(f"#{rid}" for rid in unassignable)
        super().__init__(
            "room assignment batch has no feasible solution; "
            f"unassignable reservations: {pretty}"
        )


@dataclass(frozen=True)
class MeetingRoom:
    room_id: str
    capacity: int
    equipment: tuple[str, ...]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class AssignmentRecord:
    """一条当前有效的会议室分配（批次结果与列表查询共用）。"""

    batch_id: str
    reservation_id: int
    room_id: str
    start_utc: datetime
    end_utc: datetime
    required_capacity: int
    room_capacity: int
    created_at: datetime

    @property
    def spare_seats(self) -> int:
        return self.room_capacity - self.required_capacity


@dataclass(frozen=True)
class BatchResult:
    batch_id: str
    created_at: datetime
    assignments: tuple[AssignmentRecord, ...]

    @property
    def total_spare_seats(self) -> int:
        return sum(a.spare_seats for a in self.assignments)


def normalize_equipment(tags: Iterable[str]) -> tuple[str, ...]:
    """设备标签去重 + 字典序排序（与参与者规范化规则一致）。"""
    return tuple(sorted(set(tags)))


def _row_to_room(row: sqlite3.Row) -> MeetingRoom:
    return MeetingRoom(
        room_id=row["room_id"],
        capacity=row["capacity"],
        equipment=tuple(json.loads(row["equipment_key"])),
        created_at=_parse(row["created_at"]),
        updated_at=_parse(row["updated_at"]),
    )


# ---------------------------------------------------------------- 会议室


def upsert_room(
    room_id: str,
    capacity: int,
    equipment: Iterable[str],
    db_path: str | None = None,
) -> tuple[MeetingRoom, bool]:
    """会议室 upsert，返回 ``(会议室, 是否新建)``。

    单个 ``BEGIN IMMEDIATE`` 事务内：不存在则插入（``created=True``），
    存在则整体替换容量与设备标签并刷新 ``updated_at``
    （``created_at`` 保持不变）。upsert 不回溯校验既有分配：
    已落库的分配是历史事实，仍按预约时段占用该房间。
    """
    tags = normalize_equipment(equipment)
    equipment_key = json.dumps(list(tags), ensure_ascii=False)

    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            now_s = _canon(datetime.now(timezone.utc))
            row = conn.execute(
                f"SELECT {_ROOM_COLUMNS} FROM meeting_rooms WHERE room_id = ?",
                (room_id,),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO meeting_rooms"
                    " (room_id, capacity, equipment_key, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (room_id, capacity, equipment_key, now_s, now_s),
                )
                created_at_s = now_s
                created = True
            else:
                conn.execute(
                    "UPDATE meeting_rooms"
                    " SET capacity = ?, equipment_key = ?, updated_at = ?"
                    " WHERE room_id = ?",
                    (capacity, equipment_key, now_s, room_id),
                )
                created_at_s = row["created_at"]
                created = False
            conn.execute("COMMIT")
            return MeetingRoom(
                room_id=room_id,
                capacity=capacity,
                equipment=tags,
                created_at=_parse(created_at_s),
                updated_at=_parse(now_s),
            ), created
        except BaseException:
            # 无活动事务时 rollback 为 no-op；任何失败都保证整体回滚
            conn.rollback()
            raise
    finally:
        conn.close()


def get_room(room_id: str, db_path: str | None = None) -> MeetingRoom:
    """按 room_id 读取会议室，不存在 → :class:`RoomNotFoundError`（404）。"""
    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        row = conn.execute(
            f"SELECT {_ROOM_COLUMNS} FROM meeting_rooms WHERE room_id = ?",
            (room_id,),
        ).fetchone()
        if row is None:
            raise RoomNotFoundError(room_id)
        return _row_to_room(row)
    finally:
        conn.close()


def list_rooms(
    min_capacity: int | None = None,
    required_equipment: Iterable[str] = (),
    db_path: str | None = None,
) -> list[MeetingRoom]:
    """列出会议室（按 room_id 字典序），可按最小容量与必备设备过滤。"""
    required = frozenset(required_equipment)
    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        rows = conn.execute(
            f"SELECT {_ROOM_COLUMNS} FROM meeting_rooms ORDER BY room_id"
        ).fetchall()
        rooms = [_row_to_room(row) for row in rows]
        if min_capacity is not None:
            rooms = [r for r in rooms if r.capacity >= min_capacity]
        if required:
            rooms = [r for r in rooms if required <= set(r.equipment)]
        return rooms
    finally:
        conn.close()


# -------------------------------------------------------------- 全局求解


@dataclass(frozen=True)
class _Item:
    """待分配的一条预约（求解内部表示，按 reservation_id 升序排列）。"""

    reservation_id: int
    required_capacity: int
    required_equipment: frozenset[str]
    start_s: str
    end_s: str


@dataclass(frozen=True)
class _Room:
    room_id: str
    capacity: int
    equipment: frozenset[str]


def _vec_key(choices: Iterable[str | None]) -> tuple[tuple[int, str], ...]:
    """分配向量的字典序键：已分配 ``(0, room_id)`` 小于未分配 ``(1, "")``。

    向量按预约 id 升序排列；比较即“按预约 id、room_id 字典序”，
    且未分配排在任何 room_id 之后（优先让靠前的预约获得房间）。
    """
    return tuple((0, c) if c is not None else (1, "") for c in choices)


def _solve(
    items: list[_Item],
    rooms: list[_Room],
    occupied: dict[str, list[tuple[str, str]]],
) -> tuple[tuple[str | None, ...], bool]:
    """确定性全局求解，返回 ``(分配向量, 是否整批有解)``。

    分配向量与 ``items`` 同序（预约 id 升序），每项为 room_id 或 ``None``
    （未分配）。最优性键为 ``(-已分配数, 总空余座位, 向量键)``：

    - 整批有解时，所有完备解已分配数相同，退化为
      “总空余座位最少，再按（预约 id、room_id）字典序最小”；
    - 无解时取可分配数最多的部分解，并列时总空余最少、再按向量键最小，
      其中未获分配的预约即 409 要列出的无法分配项。

    深度优先 + 剪枝枚举；枚举顺序（先试 (容量, room_id) 更紧的房间、
    最后才试不分配）只影响速度，最优性由叶子处的键比较保证。
    """
    n = len(items)
    # 每个预约的可行房间（容量 + 设备），按 (capacity, room_id) 排序：
    # 先探索更紧的 fits，让首个完备解的空余座位尽量小，加速后续剪枝
    feasible: list[list[_Room]] = []
    for it in items:
        fs = [r for r in rooms
              if r.capacity >= it.required_capacity
              and r.equipment >= it.required_equipment]
        fs.sort(key=lambda r: (r.capacity, r.room_id))
        feasible.append(fs)

    choices: list[str | None] = [None] * n
    best: tuple[tuple[object, ...], tuple[str | None, ...]] | None = None

    def overlaps(room_id: str, start_s: str, end_s: str, upto: int) -> bool:
        # 半开区间重叠：existing.start < new.end 且 new.start < existing.end
        for s, e in occupied.get(room_id, ()):
            if s < end_s and start_s < e:
                return True
        for j in range(upto):
            if choices[j] == room_id:
                oj = items[j]
                if oj.start_s < end_s and start_s < oj.end_s:
                    return True
        return False

    def dfs(i: int, spent: int, count: int) -> None:
        nonlocal best
        if best is not None:
            best_key = best[0]
            best_count = -int(best_key[0])
            best_spare = int(best_key[1])
            reachable = count + (n - i)
            if reachable < best_count:
                return  # 剩下的全部分配也追不上最优已分配数
            if reachable == best_count:
                if spent > best_spare:
                    return  # 空余座位只会继续增加，已不可能更优
                if spent == best_spare and _vec_key(choices[:i]) > best_key[2][:i]:
                    return  # 前缀已字典序落后，任何补全都无法反超
        if i == n:
            key = (-count, spent, _vec_key(choices))
            if best is None or key < best[0]:
                best = (key, tuple(choices))
            return
        it = items[i]
        for room in feasible[i]:
            if overlaps(room.room_id, it.start_s, it.end_s, i):
                continue
            choices[i] = room.room_id
            dfs(i + 1, spent + room.capacity - it.required_capacity, count + 1)
        choices[i] = None
        dfs(i + 1, spent, count)

    dfs(0, 0, 0)
    assert best is not None  # n >= 0 时至少存在全 None 的叶子
    solution = best[1]
    return solution, all(c is not None for c in solution)


# -------------------------------------------------------------- 批量分配


def _normalize_items(
    items: Iterable[tuple[int, int, Iterable[str]]],
) -> list[tuple[int, int, tuple[str, ...]]]:
    """规范化批量请求：设备去重排序，整体按预约 id 升序。"""
    norm = sorted(
        ((rid, cap, normalize_equipment(tags)) for rid, cap, tags in items),
        key=lambda t: t[0],
    )
    ids = [rid for rid, _, _ in norm]
    if len(set(ids)) != len(ids):
        # API 层以 422 拦截；核心层防御：同一预约在一批中只能出现一次
        raise ValueError("reservation ids in a batch must be unique")
    return norm


def _items_key(norm: list[tuple[int, int, tuple[str, ...]]]) -> str:
    """规范化内容的 JSON 表示，用于 batch_id 幂等比较。"""
    return json.dumps(
        [[rid, cap, list(tags)] for rid, cap, tags in norm],
        ensure_ascii=False,
    )


def _load_batch_result(
    conn: sqlite3.Connection,
    batch_pk: int,
    batch_id: str,
    created_s: str,
) -> BatchResult:
    """读取批次的**当前**分配（联表取预约时段与房间最新容量）。"""
    rows = conn.execute(
        "SELECT a.reservation_id, a.room_id, a.required_capacity, a.created_at,"
        "       r.start_utc, r.end_utc, m.capacity AS room_capacity"
        " FROM room_assignments a"
        " JOIN reservations r ON r.id = a.reservation_id"
        " JOIN meeting_rooms m ON m.room_id = a.room_id"
        " WHERE a.batch_pk = ?"
        " ORDER BY a.reservation_id",
        (batch_pk,),
    ).fetchall()
    assignments = tuple(
        AssignmentRecord(
            batch_id=batch_id,
            reservation_id=row["reservation_id"],
            room_id=row["room_id"],
            start_utc=_parse(row["start_utc"]),
            end_utc=_parse(row["end_utc"]),
            required_capacity=row["required_capacity"],
            room_capacity=row["room_capacity"],
            created_at=_parse(row["created_at"]),
        )
        for row in rows
    )
    return BatchResult(
        batch_id=batch_id,
        created_at=_parse(created_s),
        assignments=assignments,
    )


def assign_batch(
    batch_id: str,
    items: Iterable[tuple[int, int, Iterable[str]]],
    db_path: str | None = None,
) -> tuple[BatchResult, bool]:
    """批量分配会议室，返回 ``(批次结果, 是否新建)``。

    ``items`` 为 ``(reservation_id, required_capacity, required_equipment)``
    三元组。整个流程在单个 ``BEGIN IMMEDIATE`` 事务内完成，任何一步失败
    都整体回滚、已有分配保持不变：

    1. 同一 ``batch_id`` 已存在：规范化内容相同 → 幂等返回该批次当前
       分配（``created=False``）；内容不同
       → :class:`BatchContentConflictError`（409）；
    2. 任一预约不存在 → :class:`ReservationNotFoundError`（404）；
       任一预约已取消 → :class:`ReservationCanceledError`（409）；
    3. 结合**已有分配**（排除本批成员的旧分配，它们将被重新求解）
       与预约时段全局求解；整批无解
       → :class:`BatchUnsolvableError`（409，列出无法分配的预约）；
    4. 整批有解才持久化：写入批次行、删除本批成员的旧分配、
       插入新分配（同事务提交）。
    """
    norm = _normalize_items(items)
    if not norm:
        # API 层以 422 拦截；核心层防御：空批次没有求解语义
        raise ValueError("batch items must be non-empty")
    items_key = _items_key(norm)
    reservation_ids = [rid for rid, _, _ in norm]

    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        # BEGIN IMMEDIATE：事务开始即持写锁，
        # “幂等判定 + 校验 + 求解 + 持久化”对并发批次原子。
        conn.execute("BEGIN IMMEDIATE")
        try:
            now_s = _canon(datetime.now(timezone.utc))

            row = conn.execute(
                "SELECT id, batch_id, items_key, created_at"
                " FROM room_assignment_batches WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if row is not None:
                if row["items_key"] != items_key:
                    raise BatchContentConflictError(batch_id)
                result = _load_batch_result(
                    conn, row["id"], batch_id, row["created_at"])
                conn.execute("COMMIT")
                return result, False

            # 校验本批预约：必须全部存在且为 active，并取各自时段
            windows: dict[int, tuple[str, str]] = {}
            for rid in reservation_ids:
                rrow = conn.execute(
                    "SELECT id, start_utc, end_utc, status, version"
                    " FROM reservations WHERE id = ?",
                    (rid,),
                ).fetchone()
                if rrow is None:
                    raise ReservationNotFoundError(rid)
                if rrow["status"] == "canceled":
                    raise ReservationCanceledError(rid, rrow["version"])
                windows[rid] = (rrow["start_utc"], rrow["end_utc"])

            rooms = [
                _Room(
                    room_id=r["room_id"],
                    capacity=r["capacity"],
                    equipment=frozenset(json.loads(r["equipment_key"])),
                )
                for r in conn.execute(
                    "SELECT room_id, capacity, equipment_key"
                    " FROM meeting_rooms ORDER BY room_id"
                )
            ]

            # 已有分配按预约时段占用房间（仅 active 预约；排除本批成员——
            # 它们的旧分配将被本次求解替换，不构成障碍）
            placeholders = ",".join("?" for _ in reservation_ids)
            occ_rows = conn.execute(
                f"SELECT a.room_id, r.start_utc, r.end_utc"
                f" FROM room_assignments a"
                f" JOIN reservations r ON r.id = a.reservation_id"
                f" WHERE r.status = 'active'"
                f"   AND a.reservation_id NOT IN ({placeholders})",
                reservation_ids,
            ).fetchall()
            occupied: dict[str, list[tuple[str, str]]] = {}
            for o in occ_rows:
                occupied.setdefault(o["room_id"], []).append(
                    (o["start_utc"], o["end_utc"]))

            solve_items = [
                _Item(
                    reservation_id=rid,
                    required_capacity=cap,
                    required_equipment=frozenset(tags),
                    start_s=windows[rid][0],
                    end_s=windows[rid][1],
                )
                for rid, cap, tags in norm
            ]
            choices, complete = _solve(solve_items, rooms, occupied)
            if not complete:
                unassignable = [
                    solve_items[i].reservation_id
                    for i, c in enumerate(choices) if c is None
                ]
                raise BatchUnsolvableError(unassignable)

            cursor = conn.execute(
                "INSERT INTO room_assignment_batches"
                " (batch_id, items_key, created_at) VALUES (?, ?, ?)",
                (batch_id, items_key, now_s),
            )
            batch_pk = int(cursor.lastrowid)
            # 本批成员的旧分配由本次求解替换（同事务删除再插入）
            conn.execute(
                f"DELETE FROM room_assignments"
                f" WHERE reservation_id IN ({placeholders})",
                reservation_ids,
            )
            conn.executemany(
                "INSERT INTO room_assignments"
                " (batch_pk, reservation_id, room_id, required_capacity,"
                "  created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                [
                    (batch_pk, it.reservation_id, choices[i],
                     it.required_capacity, now_s)
                    for i, it in enumerate(solve_items)
                ],
            )
            result = _load_batch_result(conn, batch_pk, batch_id, now_s)
            conn.execute("COMMIT")
            return result, True
        except BaseException:
            # 无活动事务时 rollback 为 no-op；任何失败都保证整体回滚
            conn.rollback()
            raise
    finally:
        conn.close()


def get_batch(batch_id: str, db_path: str | None = None) -> BatchResult:
    """按 batch_id 读取批次及其当前分配，不存在 → :class:`BatchNotFoundError`。"""
    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        row = conn.execute(
            "SELECT id, created_at FROM room_assignment_batches"
            " WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()
        if row is None:
            raise BatchNotFoundError(batch_id)
        return _load_batch_result(conn, row["id"], batch_id, row["created_at"])
    finally:
        conn.close()


def list_assignments(
    room_id: str | None = None,
    reservation_id: int | None = None,
    db_path: str | None = None,
) -> list[AssignmentRecord]:
    """列出当前全部分配（按预约 id、room_id 升序），可按房间/预约过滤。"""
    sql = (
        "SELECT b.batch_id, a.reservation_id, a.room_id, a.required_capacity,"
        "       a.created_at, r.start_utc, r.end_utc,"
        "       m.capacity AS room_capacity"
        " FROM room_assignments a"
        " JOIN room_assignment_batches b ON b.id = a.batch_pk"
        " JOIN reservations r ON r.id = a.reservation_id"
        " JOIN meeting_rooms m ON m.room_id = a.room_id"
    )
    clauses: list[str] = []
    params: list[object] = []
    if room_id is not None:
        clauses.append("a.room_id = ?")
        params.append(room_id)
    if reservation_id is not None:
        clauses.append("a.reservation_id = ?")
        params.append(reservation_id)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY a.reservation_id, a.room_id"

    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        rows = conn.execute(sql, params).fetchall()
        return [
            AssignmentRecord(
                batch_id=row["batch_id"],
                reservation_id=row["reservation_id"],
                room_id=row["room_id"],
                start_utc=_parse(row["start_utc"]),
                end_utc=_parse(row["end_utc"]),
                required_capacity=row["required_capacity"],
                room_capacity=row["room_capacity"],
                created_at=_parse(row["created_at"]),
            )
            for row in rows
        ]
    finally:
        conn.close()


def release_assignments_for_reservations(
    conn: sqlite3.Connection,
    reservation_ids: Iterable[int],
) -> int:
    """在调用方事务内释放这些预约的会议室分配（不提交），返回释放行数。

    预约改期/取消/时段交换时调用：分配与求解时的时段绑定，
    时段变更（或预约终结）即失效，同事务释放保证已有分配之间
    永远不会出现时间重叠。
    """
    ids = tuple(reservation_ids)
    if not ids:
        return 0
    placeholders = ",".join("?" for _ in ids)
    cursor = conn.execute(
        f"DELETE FROM room_assignments WHERE reservation_id IN ({placeholders})",
        ids,
    )
    return cursor.rowcount
