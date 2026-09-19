"""预约时序依赖（dependency）核心逻辑：有向最小间隔约束的建立与变更重验。

时序依赖是两条预约之间的有向约束：

    predecessor.end_utc + min_gap_minutes <= successor.start_utc

- ``POST /api/v1/reservation-dependencies`` 在单个 ``BEGIN IMMEDIATE`` 事务内
  建边：拒绝自环（predecessor == successor）与会形成环的链路（successor
  已能沿既有依赖到达 predecessor）；相同 ``dependency_request_id`` +
  相同规范化内容（前驱、后继、间隔分钟）重试 → 幂等返回原依赖；
  同键内容不同 → :class:`DependencyContentConflictError`（409）。
  建边只校验**结构**（端点存在、无自环、无环），不要求约束当前已被满足：
  时序约束作为规则在后续变更时统一强制。
- 改期（PUT）或取消（DELETE）预约时，在**同一** ``BEGIN IMMEDIATE`` 事务内
  由 :func:`validate_reachable_dependencies` 重验该预约的**全部可达依赖**
  （从该预约出发沿依赖图双向可达的整张子图）：任一约束不满足 →
  :class:`DependencyViolationError`（API 409 + 完整链路），事务整体回滚，
  不写入部分变更；全部满足才允许提交。取消不改变起止时刻，
  因此取消路径重验的是当前已存储的约束满足状态。
"""

from __future__ import annotations

import sqlite3
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from . import db
from .reservations import (
    ReservationNotFoundError,
    _canon,
    _parse,
)

_DEPENDENCY_COLUMNS = (
    "id, dependency_request_id, predecessor_id, successor_id,"
    " min_gap_minutes, created_at"
)

_EDGE_COLUMNS = "id, predecessor_id, successor_id, min_gap_minutes"


class DependencyContentConflictError(Exception):
    """``dependency_request_id`` 已被内容不同的依赖占用。"""

    def __init__(self, existing_dependency_id: int) -> None:
        self.existing_dependency_id = existing_dependency_id
        super().__init__(
            f"dependency_request_id is already used by reservation dependency "
            f"#{existing_dependency_id} with different content"
        )


class DependencyCycleError(Exception):
    """自环或会形成环的依赖链路（依赖图必须保持有向无环）。"""


class DependencyViolationError(Exception):
    """变更后可达依赖子图中存在不满足的时序约束。"""

    def __init__(self, violations: list[dict[str, object]]) -> None:
        # 违反项形如 {"dependency_id": ..., "predecessor_id": ...,
        # "successor_id": ..., "min_gap_minutes": ...,
        # "predecessor_end_utc"/"successor_start_utc"/
        # "required_successor_start_utc": 定宽 UTC 文本,
        # "chain": [从被变更预约到该边的完整链路]}
        self.violations = violations

        def _describe(v: dict[str, object]) -> str:
            return (
                f"dependency #{v['dependency_id']} requires successor "
                f"#{v['successor_id']} to start at or after "
                f"{v['required_successor_start_utc']} (predecessor "
                f"#{v['predecessor_id']} ends at {v['predecessor_end_utc']} + "
                f"{v['min_gap_minutes']}m gap), actual start "
                f"{v['successor_start_utc']}"
            )

        pretty = "; ".join(_describe(v) for v in violations)
        super().__init__(f"reservation dependency violation: {pretty}")


@dataclass(frozen=True)
class Dependency:
    id: int
    dependency_request_id: str
    predecessor_id: int
    successor_id: int
    min_gap_minutes: int
    created_at: datetime


def _row_to_dependency(row: sqlite3.Row) -> Dependency:
    return Dependency(
        id=row["id"],
        dependency_request_id=row["dependency_request_id"],
        predecessor_id=row["predecessor_id"],
        successor_id=row["successor_id"],
        min_gap_minutes=row["min_gap_minutes"],
        created_at=_parse(row["created_at"]),
    )


def _directed_path(
    conn: sqlite3.Connection, from_id: int, to_id: int
) -> list[sqlite3.Row] | None:
    """沿依赖方向（predecessor → successor）BFS，返回 ``from_id`` 到
    ``to_id`` 的一条边路径；不可达返回 ``None``。用于建边前的成环检测。"""
    visited = {from_id}
    paths: dict[int, list[sqlite3.Row]] = {from_id: []}
    queue: deque[int] = deque([from_id])
    while queue:
        node = queue.popleft()
        rows = conn.execute(
            f"SELECT {_EDGE_COLUMNS} FROM reservation_dependencies"
            " WHERE predecessor_id = ? ORDER BY id",
            (node,),
        ).fetchall()
        for edge in rows:
            nxt = edge["successor_id"]
            if nxt in visited:
                continue
            visited.add(nxt)
            paths[nxt] = [*paths[node], edge]
            if nxt == to_id:
                return paths[nxt]
            queue.append(nxt)
    return None


def _cycle_error(
    predecessor_id: int,
    successor_id: int,
    path: list[sqlite3.Row] | None,
) -> DependencyCycleError:
    if predecessor_id == successor_id:
        return DependencyCycleError(
            "dependency cycle: predecessor and successor are the same "
            f"reservation #{predecessor_id}"
        )
    chain = [successor_id, *[edge["successor_id"] for edge in path or []]]
    pretty = " -> ".join(f"#{rid}" for rid in chain)
    return DependencyCycleError(
        f"dependency cycle: adding #{predecessor_id} -> #{successor_id} "
        f"would close a loop (existing path: {pretty})"
    )


def create_dependency(
    dependency_request_id: str,
    predecessor_id: int,
    successor_id: int,
    min_gap_minutes: int,
    db_path: str | None = None,
) -> tuple[Dependency, bool]:
    """建立时序依赖，返回 ``(依赖, 是否新建)``。

    整个流程在单个 ``BEGIN IMMEDIATE`` 事务内完成，任何一步失败都回滚：

    1. 自环（predecessor == successor）→ :class:`DependencyCycleError`；
    2. 相同 ``dependency_request_id`` 已存在：规范化内容（前驱、后继、
       间隔分钟）相同 → 幂等返回原依赖（``created=False``），不新增记录；
       内容不同 → :class:`DependencyContentConflictError`（409）；
    3. 前驱或后继预约不存在 → :class:`ReservationNotFoundError`（404）；
    4. successor 已能沿既有依赖到达 predecessor（新边会闭合成环）
       → :class:`DependencyCycleError`（409）；
    5. 否则写入新依赖。建边**不**强制约束当前已被满足：
       时序约束在后续改期/取消时统一重验（见模块 docstring）。
    """
    if predecessor_id == successor_id:
        # API 层以 422 拦截；核心层防御：自环是长度为 1 的环
        raise _cycle_error(predecessor_id, successor_id, None)

    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        # BEGIN IMMEDIATE：事务开始即持写锁，
        # “幂等判定 + 存在性 + 成环检测 + 写入”对并发请求原子。
        conn.execute("BEGIN IMMEDIATE")
        try:
            now_s = _canon(datetime.now(timezone.utc))

            row = conn.execute(
                f"SELECT {_DEPENDENCY_COLUMNS} FROM reservation_dependencies"
                " WHERE dependency_request_id = ?",
                (dependency_request_id,),
            ).fetchone()
            if row is not None:
                if (row["predecessor_id"] == predecessor_id
                        and row["successor_id"] == successor_id
                        and row["min_gap_minutes"] == min_gap_minutes):
                    conn.execute("COMMIT")
                    return _row_to_dependency(row), False
                raise DependencyContentConflictError(row["id"])

            for reservation_id in (predecessor_id, successor_id):
                exists = conn.execute(
                    "SELECT 1 FROM reservations WHERE id = ?",
                    (reservation_id,),
                ).fetchone()
                if exists is None:
                    raise ReservationNotFoundError(reservation_id)

            cycle_path = _directed_path(conn, successor_id, predecessor_id)
            if cycle_path is not None:
                raise _cycle_error(predecessor_id, successor_id, cycle_path)

            cursor = conn.execute(
                "INSERT INTO reservation_dependencies"
                " (dependency_request_id, predecessor_id, successor_id,"
                "  min_gap_minutes, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (dependency_request_id, predecessor_id, successor_id,
                 min_gap_minutes, now_s),
            )
            dependency_id = int(cursor.lastrowid)
            conn.execute("COMMIT")
            return Dependency(
                id=dependency_id,
                dependency_request_id=dependency_request_id,
                predecessor_id=predecessor_id,
                successor_id=successor_id,
                min_gap_minutes=min_gap_minutes,
                created_at=_parse(now_s),
            ), True
        except BaseException:
            # 无活动事务时 rollback 为 no-op；任何失败都保证整体回滚
            conn.rollback()
            raise
    finally:
        conn.close()


def _reachable_subgraph(
    conn: sqlite3.Connection, reservation_id: int
) -> list[tuple[sqlite3.Row, list[sqlite3.Row]]]:
    """从 ``reservation_id`` 出发沿依赖图**双向** BFS，返回可达子图中的
    全部依赖边及各自从起点出发的完整链路（边路径）。

    返回列表的每项为 ``(边, 链路)``；与被变更预约直接相连的边链路长为 1。
    依赖图在建边时保证无环，BFS 按边去重后必然终止。
    """
    visited_nodes = {reservation_id}
    visited_edges: set[int] = set()
    node_paths: dict[int, list[sqlite3.Row]] = {reservation_id: []}
    result: list[tuple[sqlite3.Row, list[sqlite3.Row]]] = []
    queue: deque[int] = deque([reservation_id])
    while queue:
        node = queue.popleft()
        rows = conn.execute(
            f"SELECT {_EDGE_COLUMNS} FROM reservation_dependencies"
            " WHERE predecessor_id = ? OR successor_id = ?"
            " ORDER BY id",
            (node, node),
        ).fetchall()
        for edge in rows:
            if edge["id"] in visited_edges:
                continue
            visited_edges.add(edge["id"])
            chain = [*node_paths[node], edge]
            result.append((edge, chain))
            other = (edge["successor_id"] if edge["predecessor_id"] == node
                     else edge["predecessor_id"])
            if other not in visited_nodes:
                visited_nodes.add(other)
                node_paths[other] = chain
                queue.append(other)
    return result


def validate_reachable_dependencies(
    conn: sqlite3.Connection,
    reservation_id: int,
    *,
    new_start_s: str | None = None,
    new_end_s: str | None = None,
) -> None:
    """重验 ``reservation_id`` 的全部可达依赖（在调用方的写事务内执行，不提交）。

    对可达子图中的每条边检查
    ``predecessor.end_utc + min_gap_minutes <= successor.start_utc``；
    ``new_start_s`` / ``new_end_s`` 为调用方即将写入的新时段（改期路径，
    两者须同时提供），未提供时使用库中已存储的起止（取消路径：
    取消不改变时段，重验的是当前满足状态）。

    任一约束不满足 → :class:`DependencyViolationError`，携带**全部**违反项
    及各自从被变更预约出发的完整链路；调用方据此回滚整个事务，
    保证不写入部分变更。全部满足（或该预约没有任何依赖）时静默返回。
    """
    edges = _reachable_subgraph(conn, reservation_id)
    if not edges:
        return

    node_ids: set[int] = {reservation_id}
    for edge, _chain in edges:
        node_ids.add(edge["predecessor_id"])
        node_ids.add(edge["successor_id"])
    placeholders = ",".join("?" for _ in node_ids)
    rows = conn.execute(
        f"SELECT id, start_utc, end_utc FROM reservations"
        f" WHERE id IN ({placeholders})",
        tuple(node_ids),
    ).fetchall()
    windows = {row["id"]: (row["start_utc"], row["end_utc"]) for row in rows}
    if new_start_s is not None and new_end_s is not None:
        # 改期：用即将提交的新时段替换该预约的库存时段后再校验，
        # 校验通过前事务尚未写入任何变更
        windows[reservation_id] = (new_start_s, new_end_s)

    violations: list[dict[str, object]] = []
    for edge, chain in edges:
        pred_end_s = windows[edge["predecessor_id"]][1]
        succ_start_s = windows[edge["successor_id"]][0]
        required_s = _canon(
            _parse(pred_end_s) + timedelta(minutes=edge["min_gap_minutes"]))
        # 定宽 UTC 文本字典序即时间序：succ.start < pred.end + gap 即违反
        if succ_start_s < required_s:
            violations.append({
                "dependency_id": edge["id"],
                "predecessor_id": edge["predecessor_id"],
                "successor_id": edge["successor_id"],
                "min_gap_minutes": edge["min_gap_minutes"],
                "predecessor_end_utc": pred_end_s,
                "successor_start_utc": succ_start_s,
                "required_successor_start_utc": required_s,
                "chain": [
                    {
                        "dependency_id": link["id"],
                        "predecessor_id": link["predecessor_id"],
                        "successor_id": link["successor_id"],
                        "min_gap_minutes": link["min_gap_minutes"],
                    }
                    for link in chain
                ],
            })
    if violations:
        violations.sort(key=lambda v: v["dependency_id"])
        raise DependencyViolationError(violations)
