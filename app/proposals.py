"""预约改期提案（reschedule proposal）核心逻辑：多人必批的改期工作流。

改期提案让一条 active 预约的改期（新时段/新参与者）必须经过一组
**必批人**（approvers）全员批准后才生效：

- ``POST /api/v1/reschedule-proposals`` 在单个 ``BEGIN IMMEDIATE`` 事务内
  建提案：提交 ``proposal_id``（客户端幂等键）、``reservation_id``、
  ``expected_version``（提案基于的预约乐观锁版本）、新时段/参与者与
  **去重排序**后的必批人列表；相同 ``proposal_id`` + 相同规范化内容重试
  → 幂等返回原提案（含其当前状态与已收票）；同键内容不同
  → :class:`ProposalContentConflictError`（409）。
  建提案只校验目标预约**存在**（404）：版本与 active 状态在最后一票
  approve 时才于同事务强制（见下），提案 pending 期间预约可正常被他人
  改期/取消。
- ``POST .../{proposal_id}/votes`` 投票：每位必批人只能 approve 或
  reject **一次**（非必批人 → :class:`ProposalVoterNotAllowedError`，
  409）；重复同票 → 幂等返回当前提案；改票
  → :class:`ProposalVoteConflictError`（409）；提案已终态
  （approved/rejected）后的新票 → :class:`ProposalNotPendingError`（409）。
  任一 **reject** 立即使提案终止为 ``rejected``（终态，不可再投票）。
- **最后一票 approve**（必批人全员批准）在**同一个** ``BEGIN IMMEDIATE``
  事务内执行改期：确认预约仍为 ``expected_version`` 且 ``active``，复用
  现有预约/保留半开区间冲突检查（排除自身）与**可达时序依赖**重验，
  然后原子替换时段与参与者占用（version+1）、追加 ``rescheduled`` 哈希链
  审计事件、释放该预约的会议室分配，并把提案置为 ``approved``。
  任一步失败（版本不符/已取消/冲突/依赖违反）→ 抛出现有异常（API 409），
  **整个事务回滚**：提案仍 ``pending``（该票也不落库）、预约不变，
  稍后可重试同一票。``BEGIN IMMEDIATE`` 串行化写者：并发的最后一票
  只有一票能提交执行，其余成为幂等重放或收到 409。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from . import audit
from . import db
from . import dependencies as dependency_core
from . import holds as hold_core
from . import reservations as reservation_core
from . import room_assignments as room_assignment_core
from .reservations import (
    Reservation,
    ReservationCanceledError,
    ReservationConflictError,
    ReservationNotFoundError,
    VersionConflictError,
    _canon,
    _parse,
)

# 提案状态：pending=待批准；approved=全员批准且改期已执行；rejected=有人否决（终态）
PROPOSAL_STATUSES = ("pending", "approved", "rejected")
# 投票取值：approve=批准；reject=否决
VOTE_DECISIONS = ("approve", "reject")

_PROPOSAL_COLUMNS = (
    "id, proposal_id, reservation_id, expected_version, start_utc, end_utc,"
    " participants_key, approvers_key, status, created_at, decided_at"
)


class ProposalNotFoundError(Exception):
    """指定 proposal_id 的改期提案不存在。"""

    def __init__(self, proposal_id: str) -> None:
        self.proposal_id = proposal_id
        super().__init__(f"reschedule proposal {proposal_id!r} not found")


class ProposalContentConflictError(Exception):
    """``proposal_id`` 已被内容不同的提案占用。"""

    def __init__(self, proposal_id: str) -> None:
        self.proposal_id = proposal_id
        super().__init__(
            f"proposal_id {proposal_id!r} is already used by a reschedule "
            "proposal with different content"
        )


class ProposalVoterNotAllowedError(Exception):
    """投票人不在提案的必批人列表中。"""

    def __init__(self, proposal_id: str, voter_id: str) -> None:
        self.proposal_id = proposal_id
        self.voter_id = voter_id
        super().__init__(
            f"voter {voter_id!r} is not a required approver of reschedule "
            f"proposal {proposal_id!r}"
        )


class ProposalVoteConflictError(Exception):
    """改票：该必批人已投过不同的票（每位必批人只能投票一次）。"""

    def __init__(self, proposal_id: str, voter_id: str, existing_decision: str) -> None:
        self.proposal_id = proposal_id
        self.voter_id = voter_id
        self.existing_decision = existing_decision
        super().__init__(
            f"voter {voter_id!r} has already voted {existing_decision!r} on "
            f"reschedule proposal {proposal_id!r}; changing the vote is not "
            "allowed"
        )


class ProposalNotPendingError(Exception):
    """提案已终态（approved/rejected），不再接受新投票。"""

    def __init__(self, proposal_id: str, status: str) -> None:
        self.proposal_id = proposal_id
        self.status = status
        super().__init__(
            f"reschedule proposal {proposal_id!r} is already {status}; "
            "no further votes are accepted"
        )


@dataclass(frozen=True)
class Vote:
    voter_id: str
    decision: str  # approve / reject
    voted_at: datetime


@dataclass(frozen=True)
class Proposal:
    id: int
    proposal_id: str
    reservation_id: int
    expected_version: int
    start_utc: datetime
    end_utc: datetime
    participants: tuple[str, ...]
    approvers: tuple[str, ...]
    status: str  # pending / approved / rejected
    created_at: datetime
    decided_at: datetime | None
    votes: tuple[Vote, ...]


def _load_votes(conn: sqlite3.Connection, proposal_pk: int) -> tuple[Vote, ...]:
    """读取提案已收票（按投票时刻、投票人字典序，确定性顺序）。"""
    rows = conn.execute(
        "SELECT voter_id, decision, voted_at FROM reschedule_proposal_votes"
        " WHERE proposal_pk = ? ORDER BY voted_at, voter_id",
        (proposal_pk,),
    ).fetchall()
    return tuple(
        Vote(
            voter_id=row["voter_id"],
            decision=row["decision"],
            voted_at=_parse(row["voted_at"]),
        )
        for row in rows
    )


def _row_to_proposal(row: sqlite3.Row, votes: tuple[Vote, ...]) -> Proposal:
    decided_raw = row["decided_at"]
    return Proposal(
        id=row["id"],
        proposal_id=row["proposal_id"],
        reservation_id=row["reservation_id"],
        expected_version=row["expected_version"],
        start_utc=_parse(row["start_utc"]),
        end_utc=_parse(row["end_utc"]),
        participants=tuple(json.loads(row["participants_key"])),
        approvers=tuple(json.loads(row["approvers_key"])),
        status=row["status"],
        created_at=_parse(row["created_at"]),
        decided_at=_parse(decided_raw) if decided_raw else None,
        votes=votes,
    )


def _load_proposal(conn: sqlite3.Connection, proposal_pk: int) -> Proposal:
    """在调用方事务内按主键读取提案及其已收票。"""
    row = conn.execute(
        f"SELECT {_PROPOSAL_COLUMNS} FROM reschedule_proposals WHERE id = ?",
        (proposal_pk,),
    ).fetchone()
    return _row_to_proposal(row, _load_votes(conn, proposal_pk))


def _load_reservation(conn: sqlite3.Connection, reservation_id: int) -> Reservation:
    """读取提案目标预约（外键保证存在；预约只软删除不物理删除）。"""
    reservation = reservation_core.load_reservation(conn, reservation_id)
    assert reservation is not None  # 提案创建时已校验存在，且预约行永不物理删除
    return reservation


def create_proposal(
    proposal_id: str,
    reservation_id: int,
    expected_version: int,
    start_utc: datetime,
    end_utc: datetime,
    participant_ids: Iterable[str],
    approver_ids: Iterable[str],
    db_path: str | None = None,
) -> tuple[Proposal, Reservation, bool]:
    """创建改期提案，返回 ``(提案, 目标预约当前状态, 是否新建)``。

    整个流程在单个 ``BEGIN IMMEDIATE`` 事务内完成：

    1. 相同 ``proposal_id`` 已存在：规范化内容（目标预约、期望版本、
       新起止、去重排序参与者、去重排序必批人）相同 → 幂等返回原提案
       （``created=False``，含其当前状态与已收票），不新增记录；
       内容不同 → :class:`ProposalContentConflictError`（409）；
    2. 目标预约不存在 → :class:`ReservationNotFoundError`（404）；
    3. 否则写入 ``pending`` 提案。建提案**不**校验版本与 active 状态：
       这些在最后一票 approve 时于同一事务强制（见
       :func:`vote_on_proposal`），提案 pending 期间预约可正常变更。
    """
    participants = reservation_core.normalize_participants(participant_ids)
    approvers = reservation_core.normalize_participants(approver_ids)
    start_s = _canon(start_utc)
    end_s = _canon(end_utc)
    participants_key = json.dumps(list(participants), ensure_ascii=False)
    approvers_key = json.dumps(list(approvers), ensure_ascii=False)

    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        # BEGIN IMMEDIATE：事务开始即持写锁，
        # “幂等判定 + 存在性校验 + 写入”对并发请求原子。
        conn.execute("BEGIN IMMEDIATE")
        try:
            now_s = _canon(datetime.now(timezone.utc))

            row = conn.execute(
                f"SELECT {_PROPOSAL_COLUMNS} FROM reschedule_proposals"
                " WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
            if row is not None:
                if (row["reservation_id"] == reservation_id
                        and row["expected_version"] == expected_version
                        and row["start_utc"] == start_s
                        and row["end_utc"] == end_s
                        and row["participants_key"] == participants_key
                        and row["approvers_key"] == approvers_key):
                    proposal = _load_proposal(conn, row["id"])
                    reservation = _load_reservation(conn, reservation_id)
                    conn.execute("COMMIT")
                    return proposal, reservation, False
                raise ProposalContentConflictError(proposal_id)

            exists = conn.execute(
                "SELECT 1 FROM reservations WHERE id = ?",
                (reservation_id,),
            ).fetchone()
            if exists is None:
                raise ReservationNotFoundError(reservation_id)

            cursor = conn.execute(
                "INSERT INTO reschedule_proposals"
                " (proposal_id, reservation_id, expected_version, start_utc,"
                "  end_utc, participants_key, approvers_key, status, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
                (proposal_id, reservation_id, expected_version, start_s, end_s,
                 participants_key, approvers_key, now_s),
            )
            proposal_pk = int(cursor.lastrowid)
            proposal = _load_proposal(conn, proposal_pk)
            reservation = _load_reservation(conn, reservation_id)
            conn.execute("COMMIT")
            return proposal, reservation, True
        except BaseException:
            # 无活动事务时 rollback 为 no-op；任何失败都保证整体回滚
            conn.rollback()
            raise
    finally:
        conn.close()


def get_proposal(
    proposal_id: str,
    db_path: str | None = None,
) -> tuple[Proposal, Reservation]:
    """按 ``proposal_id`` 查询提案（含已收票）与目标预约当前状态。

    提案不存在 → :class:`ProposalNotFoundError`（API 层映射为 404）。
    """
    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        row = conn.execute(
            "SELECT id FROM reschedule_proposals WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        if row is None:
            raise ProposalNotFoundError(proposal_id)
        proposal = _load_proposal(conn, row["id"])
        reservation = _load_reservation(conn, proposal.reservation_id)
        return proposal, reservation
    finally:
        conn.close()


def _execute_reschedule(
    conn: sqlite3.Connection,
    proposal_row: sqlite3.Row,
    participants: tuple[str, ...],
    now_s: str,
) -> None:
    """最后一票 approve：在调用方事务内校验并执行改期（不自行提交）。

    与 :func:`reservations.reschedule_reservation` 同语义、同顺序：
    状态/版本校验 → 预约与未过期保留的半开区间冲突检查（排除自身）→
    可达时序依赖重验 → 释放会议室分配 → 原子替换时段与参与者占用
    （version+1）→ 追加 ``rescheduled`` 哈希链审计事件。

    任一步失败抛出现有异常（:class:`ReservationCanceledError` /
    :class:`VersionConflictError` / :class:`ReservationConflictError` /
    :class:`dependencies.DependencyViolationError`，API 均映射 409），
    由调用方回滚整个事务：提案仍 pending、本票不落库、预约不变。
    """
    reservation_id = proposal_row["reservation_id"]
    start_s = proposal_row["start_utc"]
    end_s = proposal_row["end_utc"]

    rrow = conn.execute(
        f"SELECT {reservation_core._RESERVATION_COLUMNS} FROM reservations"
        " WHERE id = ?",
        (reservation_id,),
    ).fetchone()
    if rrow is None:
        # 外键保证存在（预约只软删除）；防御性分支
        raise ReservationNotFoundError(reservation_id)
    if rrow["status"] == "canceled":
        # 取消是终态：提案执行不会使其复活
        raise ReservationCanceledError(reservation_id, rrow["version"])
    if rrow["version"] != proposal_row["expected_version"]:
        # 提案期间预约已被他人变更：乐观锁失败，提案保持 pending
        raise VersionConflictError(reservation_id, rrow["version"])

    # 半开区间重叠（同改期路径），排除自身旧占用；
    # 未过期的临时保留同样占用时段
    conflicts = reservation_core.collect_reservation_conflicts(
        conn, participants, start_s, end_s,
        exclude_reservation_ids=(reservation_id,))
    conflicts.extend(hold_core.collect_hold_conflicts(
        conn, participants, start_s, end_s, now_s))
    if conflicts:
        raise ReservationConflictError(conflicts)

    # 时序依赖重验：新时段下该预约的全部可达依赖必须仍然满足，
    # 否则抛出 DependencyViolationError（此时尚未写入任何变更）
    dependency_core.validate_reachable_dependencies(
        conn, reservation_id, new_start_s=start_s, new_end_s=end_s)

    # 会议室分配与求解时的时段绑定：改期即失效，同事务释放
    room_assignment_core.release_assignments_for_reservations(
        conn, (reservation_id,))

    participants_key = json.dumps(list(participants), ensure_ascii=False)
    conn.execute(
        "UPDATE reservations"
        " SET start_utc = ?, end_utc = ?, participants_key = ?,"
        "     version = version + 1"
        " WHERE id = ?",
        (start_s, end_s, participants_key, reservation_id),
    )
    # 参与者集合可能变化：先删除自身全部旧占用，再插入新占用；
    # 与 UPDATE 同事务提交，失败即整体回滚
    conn.execute(
        "DELETE FROM reservation_participants WHERE reservation_id = ?",
        (reservation_id,),
    )
    conn.executemany(
        "INSERT INTO reservation_participants"
        " (reservation_id, participant_id, start_utc, end_utc)"
        " VALUES (?, ?, ?, ?)",
        [(reservation_id, pid, start_s, end_s) for pid in participants],
    )
    # 同事务追加 rescheduled 审计事件：快照为改期后的新状态，
    # request_id / created_at 沿用原值；now_s 取事务入口时刻
    snapshot = audit.build_snapshot(
        reservation_id=reservation_id,
        request_id=rrow["request_id"],
        start_utc=start_s,
        end_utc=end_s,
        participants=participants,
        version=rrow["version"] + 1,
        status="active",
        canceled_at=None,
        created_at=rrow["created_at"],
    )
    audit.append_event(
        conn,
        reservation_id=reservation_id,
        action="rescheduled",
        timestamp=now_s,
        snapshot=snapshot,
    )


def vote_on_proposal(
    proposal_id: str,
    voter_id: str,
    decision: str,
    db_path: str | None = None,
) -> tuple[Proposal, Reservation, bool]:
    """对提案投票，返回 ``(提案, 目标预约当前状态, 是否本次新投票)``。

    整个流程在单个 ``BEGIN IMMEDIATE`` 事务内完成，任何一步失败都回滚：

    1. 提案不存在 → :class:`ProposalNotFoundError`（404）；
    2. 投票人不在必批人列表 → :class:`ProposalVoterNotAllowedError`（409）；
    3. 该必批人已投票：与本次相同 → 幂等返回当前提案（``recorded=False``）；
       不同（改票）→ :class:`ProposalVoteConflictError`（409）；
    4. 提案已终态（approved/rejected）→ :class:`ProposalNotPendingError`
       （409），不再接受新票；
    5. 记录新票。**reject** 立即使提案终止为 ``rejected``；**approve**
       若使必批人全员批准（最后一票），在同一事务内执行改期
       （:func:`_execute_reschedule`）并把提案置为 ``approved``——
       任一步失败整个事务回滚（含本票），提案仍 ``pending``、预约不变。
    """
    if decision not in VOTE_DECISIONS:
        # API 层以 422 拦截；核心层防御：只接受 approve / reject
        raise ValueError(f"decision must be one of {VOTE_DECISIONS}")

    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        # BEGIN IMMEDIATE：事务开始即持写锁，“读提案 + 记票 + 可能执行改期”
        # 对并发请求原子；并发的最后一票只有一票能提交执行。
        conn.execute("BEGIN IMMEDIATE")
        try:
            now_s = _canon(datetime.now(timezone.utc))
            # 事务内先惰性过期到期保留：随后的冲突检查只面对有效占用
            hold_core.expire_due_holds(conn, now_s)

            row = conn.execute(
                f"SELECT {_PROPOSAL_COLUMNS} FROM reschedule_proposals"
                " WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
            if row is None:
                raise ProposalNotFoundError(proposal_id)
            proposal_pk = row["id"]
            approvers = tuple(json.loads(row["approvers_key"]))
            if voter_id not in approvers:
                raise ProposalVoterNotAllowedError(proposal_id, voter_id)

            existing = conn.execute(
                "SELECT decision FROM reschedule_proposal_votes"
                " WHERE proposal_pk = ? AND voter_id = ?",
                (proposal_pk, voter_id),
            ).fetchone()
            if existing is not None:
                if existing["decision"] == decision:
                    # 重复同票：幂等返回当前提案（终态与否均如此），不重复记票
                    proposal = _load_proposal(conn, proposal_pk)
                    reservation = _load_reservation(conn, row["reservation_id"])
                    conn.execute("COMMIT")
                    return proposal, reservation, False
                # 每位必批人只能投票一次：改票拒绝
                raise ProposalVoteConflictError(
                    proposal_id, voter_id, existing["decision"])

            if row["status"] != "pending":
                # 提案已终态：不再接受新投票
                raise ProposalNotPendingError(proposal_id, row["status"])

            conn.execute(
                "INSERT INTO reschedule_proposal_votes"
                " (proposal_pk, voter_id, decision, voted_at)"
                " VALUES (?, ?, ?, ?)",
                (proposal_pk, voter_id, decision, now_s),
            )

            if decision == "reject":
                # 任一 reject 使提案终止（终态）
                conn.execute(
                    "UPDATE reschedule_proposals"
                    " SET status = 'rejected', decided_at = ? WHERE id = ?",
                    (now_s, proposal_pk),
                )
            else:
                approved = conn.execute(
                    "SELECT COUNT(*) AS n FROM reschedule_proposal_votes"
                    " WHERE proposal_pk = ? AND decision = 'approve'",
                    (proposal_pk,),
                ).fetchone()["n"]
                if approved == len(approvers):
                    # 最后一票 approve：同事务确认版本/状态、复用冲突与
                    # 依赖校验后执行改期；任一步失败整体回滚（含本票），
                    # 提案仍 pending、预约不变
                    participants = tuple(json.loads(row["participants_key"]))
                    _execute_reschedule(conn, row, participants, now_s)
                    conn.execute(
                        "UPDATE reschedule_proposals"
                        " SET status = 'approved', decided_at = ? WHERE id = ?",
                        (now_s, proposal_pk),
                    )

            proposal = _load_proposal(conn, proposal_pk)
            reservation = _load_reservation(conn, row["reservation_id"])
            conn.execute("COMMIT")
            return proposal, reservation, True
        except BaseException:
            # 无活动事务时 rollback 为 no-op；任何失败都保证整体回滚
            conn.rollback()
            raise
    finally:
        conn.close()
