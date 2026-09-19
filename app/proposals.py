"""预约改期提案（reschedule proposal）核心逻辑：必批人投票驱动的原子改期。

改期提案把一次预约改期拆成“提案 + 必批人投票”两步：

- ``POST /api/v1/reschedule-proposals`` 在单个 ``BEGIN IMMEDIATE`` 事务内
  创建提案：提交 ``proposal_id``（客户端幂等键）、``reservation_id``、
  ``expected_version``（提案基于的预约乐观锁版本）、新时段/参与者与
  去重排序后的必批人列表。相同 ``proposal_id`` + 相同规范化内容重试
  → 幂等返回原提案；同键内容不同
  → :class:`ProposalContentConflictError`（409）。
  创建只要求目标预约存在（404）；版本/状态/冲突校验**推迟到最后一票
  approve**——投票期间预约被他人改动与提案内容本身非法走同一条 409
  路径，提案在投票期间不锁定预约。
- ``POST .../{proposal_id}/votes`` 投票：每位必批人只能 approve 或
  reject **一次**——重复同票幂等返回当前提案，改票
  → :class:`VoteConflictError`（409）；非必批人
  → :class:`NotApproverError`（409）；提案已终止（approved/rejected）
  → :class:`ProposalClosedError`（409）。任一 reject 立即把提案置为
  rejected（终止），预约不做任何变更。
- **最后一票 approve**（必批人全部 approve）在同一 ``BEGIN IMMEDIATE``
  事务内完成改期：确认预约仍为 ``expected_version`` 且 active，复用现有
  进行中预约/未过期保留的半开区间冲突检查与可达时序依赖重验
  （:mod:`app.dependencies`），随后执行改期、追加 ``rescheduled``
  哈希链审计事件（:mod:`app.audit`）并释放会议室分配
  （:mod:`app.room_assignments`），最后把提案置为 approved。
  **任一步失败整体回滚**：提案仍 pending、预约不变、该票也不落库
  （条件满足后重试同一票即可）。``BEGIN IMMEDIATE`` 写锁把并发最后投票
  串行化：改期只会提交一次。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from . import audit
from . import db
from .reservations import (
    _RESERVATION_COLUMNS,
    Reservation,
    ReservationCanceledError,
    ReservationConflictError,
    ReservationNotFoundError,
    VersionConflictError,
    _canon,
    _parse,
    collect_reservation_conflicts,
    load_reservation,
    normalize_participants,
)

_PROPOSAL_COLUMNS = (
    "id, proposal_id, reservation_id, expected_version, start_utc, end_utc,"
    " participants_key, approvers_key, status, created_at, resolved_at"
)

DECISIONS = ("approve", "reject")


class ProposalNotFoundError(Exception):
    """指定 ``proposal_id`` 的改期提案不存在。"""

    def __init__(self, proposal_id: str) -> None:
        self.proposal_id = proposal_id
        super().__init__(f"reschedule proposal {proposal_id!r} not found")


class ProposalContentConflictError(Exception):
    """``proposal_id`` 已被内容不同的改期提案占用。"""

    def __init__(self, proposal_id: str) -> None:
        self.existing_proposal_id = proposal_id
        super().__init__(
            f"proposal_id is already used by reschedule proposal "
            f"{proposal_id!r} with different content"
        )


class ProposalClosedError(Exception):
    """提案已终止（approved / rejected）：不再接受新投票。"""

    def __init__(self, proposal_id: str, status: str) -> None:
        self.proposal_id = proposal_id
        self.status = status
        super().__init__(
            f"reschedule proposal {proposal_id!r} is already {status}"
        )


class NotApproverError(Exception):
    """投票人不在提案的必批人列表内。"""

    def __init__(self, proposal_id: str, approver_id: str) -> None:
        self.proposal_id = proposal_id
        self.approver_id = approver_id
        super().__init__(
            f"{approver_id!r} is not a required approver of reschedule "
            f"proposal {proposal_id!r}"
        )


class VoteConflictError(Exception):
    """该必批人已投过不同的票：每位必批人只能 approve 或 reject 一次。"""

    def __init__(self, proposal_id: str, approver_id: str,
                 existing_decision: str) -> None:
        self.proposal_id = proposal_id
        self.approver_id = approver_id
        self.existing_decision = existing_decision
        super().__init__(
            f"approver {approver_id!r} has already voted "
            f"{existing_decision!r} on reschedule proposal {proposal_id!r}"
        )


@dataclass(frozen=True)
class ProposalVote:
    approver_id: str
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
    created_at: datetime
    status: str = "pending"  # pending / approved / rejected
    resolved_at: datetime | None = None
    votes: tuple[ProposalVote, ...] = ()


def _load_votes(
    conn: sqlite3.Connection, proposal_pk: int
) -> tuple[ProposalVote, ...]:
    """读取提案的全部投票（按投票时刻、必批人 id 升序，保证确定性顺序）。"""
    rows = conn.execute(
        "SELECT approver_id, decision, voted_at"
        " FROM reschedule_proposal_votes WHERE proposal_pk = ?"
        " ORDER BY voted_at, approver_id",
        (proposal_pk,),
    ).fetchall()
    return tuple(
        ProposalVote(
            approver_id=row["approver_id"],
            decision=row["decision"],
            voted_at=_parse(row["voted_at"]),
        )
        for row in rows
    )


def _row_to_proposal(
    row: sqlite3.Row, votes: tuple[ProposalVote, ...]
) -> Proposal:
    resolved_raw = row["resolved_at"]
    return Proposal(
        id=row["id"],
        proposal_id=row["proposal_id"],
        reservation_id=row["reservation_id"],
        expected_version=row["expected_version"],
        start_utc=_parse(row["start_utc"]),
        end_utc=_parse(row["end_utc"]),
        participants=tuple(json.loads(row["participants_key"])),
        approvers=tuple(json.loads(row["approvers_key"])),
        created_at=_parse(row["created_at"]),
        status=row["status"],
        resolved_at=_parse(resolved_raw) if resolved_raw else None,
        votes=votes,
    )


def _load_proposal(conn: sqlite3.Connection, proposal_pk: int) -> Proposal:
    """在调用方事务内按主键读取提案（含投票）；调用方保证行存在。"""
    row = conn.execute(
        f"SELECT {_PROPOSAL_COLUMNS} FROM reschedule_proposals WHERE id = ?",
        (proposal_pk,),
    ).fetchone()
    return _row_to_proposal(row, _load_votes(conn, proposal_pk))


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

    整个流程在单个 ``BEGIN IMMEDIATE`` 事务内完成，任何一步失败都回滚：

    1. 相同 ``proposal_id`` 已存在：规范化内容（目标预约、期望版本、起止、
       去重排序参与者、去重排序必批人）相同 → 幂等返回原提案
       （``created=False``），不新增记录；内容不同
       → :class:`ProposalContentConflictError`（409）；
    2. 目标预约不存在 → :class:`ReservationNotFoundError`（404）；
    3. 否则写入 ``pending`` 提案。创建**不**校验版本/状态/冲突：
       这些校验在最后一票 approve 时于同一事务内执行
       （见 :func:`vote_proposal`），提案在投票期间不锁定预约。
    """
    participants = normalize_participants(participant_ids)
    approvers = normalize_participants(approver_ids)  # 同一去重排序规范化
    if not approvers:
        # API 层以 422 拦截；核心层防御：没有必批人的提案永远无法通过
        raise ValueError("approver_ids must be non-empty")
    start_s = _canon(start_utc)
    end_s = _canon(end_utc)
    participants_key = json.dumps(list(participants), ensure_ascii=False)
    approvers_key = json.dumps(list(approvers), ensure_ascii=False)

    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        # BEGIN IMMEDIATE：事务开始即持写锁，
        # “幂等判定 + 存在性 + 写入”对并发请求原子。
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
                    proposal = _row_to_proposal(row, _load_votes(conn, row["id"]))
                    reservation = load_reservation(conn, reservation_id)
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
                " (proposal_id, reservation_id, expected_version,"
                "  start_utc, end_utc, participants_key, approvers_key,"
                "  status, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
                (proposal_id, reservation_id, expected_version,
                 start_s, end_s, participants_key, approvers_key, now_s),
            )
            proposal_pk = int(cursor.lastrowid)
            proposal = Proposal(
                id=proposal_pk,
                proposal_id=proposal_id,
                reservation_id=reservation_id,
                expected_version=expected_version,
                start_utc=_parse(start_s),
                end_utc=_parse(end_s),
                participants=participants,
                approvers=approvers,
                created_at=_parse(now_s),
            )
            reservation = load_reservation(conn, reservation_id)
            conn.execute("COMMIT")
            return proposal, reservation, True
        except BaseException:
            # 无活动事务时 rollback 为 no-op；任何失败都保证整体回滚
            conn.rollback()
            raise
    finally:
        conn.close()


def vote_proposal(
    proposal_id: str,
    approver_id: str,
    decision: str,
    db_path: str | None = None,
) -> tuple[Proposal, Reservation, bool]:
    """对提案投一票，返回 ``(提案, 目标预约当前状态, 是否新投票)``。

    整个流程在单个 ``BEGIN IMMEDIATE`` 事务内完成，任何一步失败都回滚：

    1. 提案不存在 → :class:`ProposalNotFoundError`（404）；
    2. 投票人不在必批人列表 → :class:`NotApproverError`（409）；
    3. 该必批人已投票：与本次相同 → 幂等返回当前提案
       （``recorded=False``）；不同（改票）
       → :class:`VoteConflictError`（409）；
    4. 提案已终止（approved/rejected）→ :class:`ProposalClosedError`（409）；
    5. 写入投票。reject → 同事务把提案置为 rejected（任一否决即终止），
       预约不做任何变更；
    6. approve 且必批人已**全部** approve（最后一票）→ 同一事务内：
       预约仍为 ``expected_version`` 且 active（否则 409）、复用进行中
       预约/未过期保留的半开区间冲突检查与可达时序依赖重验（否则 409）、
       执行改期 + ``rescheduled`` 哈希链审计事件 + 释放会议室分配，
       提案置为 approved。**任一步失败整体回滚**：提案仍 pending、
       预约不变、该票也不落库（条件满足后重试同一票即可）。
    """
    if decision not in DECISIONS:
        # API 层以 422 拦截；核心层防御：只接受 approve / reject
        raise ValueError(f"unknown vote decision: {decision!r}")

    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        # BEGIN IMMEDIATE：事务开始即持写锁，“读提案/投票状态 + 写投票 +
        # （最后一票时）校验 + 改期”对并发请求原子——并发最后投票被串行化，
        # 改期只会提交一次。
        conn.execute("BEGIN IMMEDIATE")
        try:
            now_s = _canon(datetime.now(timezone.utc))
            # 事务入口惰性过期到期保留：最后一票的冲突检查只面对有效占用
            # （惰性导入避免与 holds 模块循环导入）
            from . import holds as hold_core
            hold_core.expire_due_holds(conn, now_s)

            row = conn.execute(
                f"SELECT {_PROPOSAL_COLUMNS} FROM reschedule_proposals"
                " WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
            if row is None:
                raise ProposalNotFoundError(proposal_id)

            approvers = tuple(json.loads(row["approvers_key"]))
            if approver_id not in approvers:
                raise NotApproverError(proposal_id, approver_id)

            existing_vote = conn.execute(
                "SELECT decision FROM reschedule_proposal_votes"
                " WHERE proposal_pk = ? AND approver_id = ?",
                (row["id"], approver_id),
            ).fetchone()
            if existing_vote is not None:
                if existing_vote["decision"] == decision:
                    # 重复同票：幂等返回当前提案，不重复写库
                    proposal = _row_to_proposal(row, _load_votes(conn, row["id"]))
                    reservation = load_reservation(conn, row["reservation_id"])
                    conn.execute("COMMIT")
                    return proposal, reservation, False
                # 改票：每位必批人只能 approve 或 reject 一次
                raise VoteConflictError(
                    proposal_id, approver_id, existing_vote["decision"])

            if row["status"] != "pending":
                raise ProposalClosedError(proposal_id, row["status"])

            conn.execute(
                "INSERT INTO reschedule_proposal_votes"
                " (proposal_pk, approver_id, decision, voted_at)"
                " VALUES (?, ?, ?, ?)",
                (row["id"], approver_id, decision, now_s),
            )

            if decision == "reject":
                # 任一否决即终止：提案置为 rejected，预约不做任何变更
                conn.execute(
                    "UPDATE reschedule_proposals"
                    " SET status = 'rejected', resolved_at = ? WHERE id = ?",
                    (now_s, row["id"]),
                )
                proposal = _load_proposal(conn, row["id"])
                reservation = load_reservation(conn, row["reservation_id"])
                conn.execute("COMMIT")
                return proposal, reservation, True

            approved = conn.execute(
                "SELECT COUNT(*) AS n FROM reschedule_proposal_votes"
                " WHERE proposal_pk = ? AND decision = 'approve'",
                (row["id"],),
            ).fetchone()["n"]
            if approved < len(approvers):
                # 尚有必批人未同意：提案继续 pending
                proposal = _load_proposal(conn, row["id"])
                reservation = load_reservation(conn, row["reservation_id"])
                conn.execute("COMMIT")
                return proposal, reservation, True

            # ---------------- 最后一票 approve：同事务校验并执行改期 ----------------
            reservation_id = row["reservation_id"]
            rrow = conn.execute(
                f"SELECT {_RESERVATION_COLUMNS} FROM reservations WHERE id = ?",
                (reservation_id,),
            ).fetchone()
            if rrow is None:
                raise ReservationNotFoundError(reservation_id)
            if rrow["status"] == "canceled":
                # 取消是终态：改期不会使其复活；回滚后提案仍 pending
                raise ReservationCanceledError(reservation_id, rrow["version"])
            if row["expected_version"] != rrow["version"]:
                # 投票期间预约已被他人改动：乐观锁拦截，回滚后提案仍 pending
                raise VersionConflictError(reservation_id, rrow["version"])

            participants = tuple(json.loads(row["participants_key"]))
            start_s = row["start_utc"]
            end_s = row["end_utc"]
            # 半开区间冲突（排除自身旧占用）+ 未过期 pending 保留
            conflicts = collect_reservation_conflicts(
                conn, participants, start_s, end_s,
                exclude_reservation_ids=(reservation_id,))
            conflicts.extend(hold_core.collect_hold_conflicts(
                conn, participants, start_s, end_s, now_s))
            if conflicts:
                raise ReservationConflictError(conflicts)

            # 时序依赖重验：新时段下全部可达依赖必须满足
            # （惰性导入避免与 dependencies 模块循环导入）
            from . import dependencies as dependency_core
            dependency_core.validate_reachable_dependencies(
                conn, reservation_id, new_start_s=start_s, new_end_s=end_s)

            # 会议室分配与求解时的时段绑定：改期即失效，同事务释放
            # （惰性导入避免与 room_assignments 模块循环导入）
            from . import room_assignments as room_assignment_core
            room_assignment_core.release_assignments_for_reservations(
                conn, (reservation_id,))

            # 执行改期：更新预约行并整体替换参与者占用（与 PUT 改期一致）
            conn.execute(
                "UPDATE reservations"
                " SET start_utc = ?, end_utc = ?, participants_key = ?,"
                "     version = version + 1"
                " WHERE id = ?",
                (start_s, end_s, row["participants_key"], reservation_id),
            )
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
            # 同事务追加 rescheduled 审计事件：快照为改期后的新状态
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
            # 提案终止：approved
            conn.execute(
                "UPDATE reschedule_proposals"
                " SET status = 'approved', resolved_at = ? WHERE id = ?",
                (now_s, row["id"]),
            )
            proposal = _load_proposal(conn, row["id"])
            reservation = load_reservation(conn, reservation_id)
            conn.execute("COMMIT")
            return proposal, reservation, True
        except BaseException:
            # 无活动事务时 rollback 为 no-op；任何失败都保证整体回滚
            # （最后一票失败时该票也不落库，提案仍 pending，预约不变）
            conn.rollback()
            raise
    finally:
        conn.close()


def get_proposal(
    proposal_id: str,
    db_path: str | None = None,
) -> tuple[Proposal, Reservation]:
    """按 ``proposal_id`` 读取提案（含全部投票）与目标预约当前状态。

    提案不存在 → :class:`ProposalNotFoundError`（404）。
    """
    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        row = conn.execute(
            f"SELECT {_PROPOSAL_COLUMNS} FROM reschedule_proposals"
            " WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        if row is None:
            raise ProposalNotFoundError(proposal_id)
        proposal = _row_to_proposal(row, _load_votes(conn, row["id"]))
        reservation = load_reservation(conn, row["reservation_id"])
        if reservation is None:
            # 关联预约行缺失属于数据损坏（外键约束下不应发生）
            raise ReservationNotFoundError(row["reservation_id"])
        return proposal, reservation
    finally:
        conn.close()
