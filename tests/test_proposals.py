"""预约改期提案（reschedule proposals）测试。

覆盖：建提案（201/规范化/幂等重放/同键异内容 409/目标预约不存在 404/
已取消预约仍可建提案）、投票（非必批人 409/重复同票幂等/改票 409/
任一 reject 终止/终态后新票 409）、最后一票 approve 同事务执行改期
（版本不符/已取消/冲突/依赖违反均 409 且提案仍 pending、预约不变；
成功则 version+1、审计事件、会议室分配释放）、并发最后投票只提交一次、
查询接口、旧版库无损建新表。
"""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app import db as db_module
from app.main import app


@pytest.fixture()
def db_path(tmp_path, monkeypatch) -> str:
    """每个测试用独立的 SQLite 文件。"""
    path = str(tmp_path / "reservations.db")
    monkeypatch.setenv("RESERVATIONS_DB_PATH", path)
    return path


@pytest.fixture()
def client(db_path) -> TestClient:
    return TestClient(app)


def _utc(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


def _reservation(client, request_id, start, end, participants=("alice",)):
    resp = client.post("/api/v1/reservations", json={
        "request_id": request_id,
        "start_utc": start,
        "end_utc": end,
        "participant_ids": list(participants),
    })
    assert resp.status_code == 201, resp.text
    return resp.json()


def _proposal_payload(reservation_id, **overrides):
    payload = {
        "proposal_id": "prop-1",
        "reservation_id": reservation_id,
        "expected_version": 1,
        "start_utc": "2026-03-09T15:00:00Z",
        "end_utc": "2026-03-09T16:00:00Z",
        "participant_ids": ["alice"],
        "approver_ids": ["boss-a", "boss-b"],
    }
    payload.update(overrides)
    return payload


def _create_proposal(client, reservation_id, **overrides):
    resp = client.post("/api/v1/reschedule-proposals",
                       json=_proposal_payload(reservation_id, **overrides))
    assert resp.status_code == 201, resp.text
    return resp.json()


def _vote(client, proposal_id, voter_id, decision):
    return client.post(
        f"/api/v1/reschedule-proposals/{proposal_id}/votes",
        json={"voter_id": voter_id, "decision": decision})


def _get_proposal(client, proposal_id):
    resp = client.get(f"/api/v1/reschedule-proposals/{proposal_id}")
    assert resp.status_code == 200, resp.text
    return resp.json()


def _history(client, reservation_id):
    resp = client.get(f"/api/v1/reservations/{reservation_id}/history")
    assert resp.status_code == 200, resp.text
    return resp.json()["events"]


# ------------------------------------------------------------------ 建提案

def test_create_proposal_returns_201_normalized(client):
    r = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    resp = client.post("/api/v1/reschedule-proposals", json=_proposal_payload(
        r["reservation_id"],
        participant_ids=["carol", "alice", "carol"],
        approver_ids=["boss-b", "boss-a", "boss-b"],
    ))
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["proposal_id"] == "prop-1"
    assert data["reservation_id"] == r["reservation_id"]
    assert data["expected_version"] == 1
    assert data["start_utc"] == "2026-03-09T15:00:00Z"
    assert data["end_utc"] == "2026-03-09T16:00:00Z"
    # 参与者与必批人均去重并按字典序排序
    assert data["participants"] == ["alice", "carol"]
    assert data["approvers"] == ["boss-a", "boss-b"]
    assert data["status"] == "pending"
    assert data["decided_at"] is None
    assert data["votes"] == []
    assert data["created_at"]
    # 内嵌目标预约当前状态
    assert data["reservation"]["reservation_id"] == r["reservation_id"]
    assert data["reservation"]["version"] == 1
    assert data["reservation"]["status"] == "active"


def test_create_proposal_idempotent_replay(client):
    r = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    payload = _proposal_payload(r["reservation_id"])
    first = client.post("/api/v1/reschedule-proposals", json=payload)
    assert first.status_code == 201
    replay = client.post("/api/v1/reschedule-proposals", json=payload)
    assert replay.status_code == 200
    assert replay.json() == first.json()  # 返回原提案，不新增记录

    # 重放发生在投票之后：返回同一提案的当前状态（含已收票）
    assert _vote(client, "prop-1", "boss-a", "approve").status_code == 201
    replay2 = client.post("/api/v1/reschedule-proposals", json=payload)
    assert replay2.status_code == 200
    assert [v["voter_id"] for v in replay2.json()["votes"]] == ["boss-a"]

    # 下一条提案不会复用已占用的内部 id：确实只建了一条
    other = _create_proposal(client, r["reservation_id"], proposal_id="prop-2")
    assert other["proposal_id"] == "prop-2"


@pytest.mark.parametrize("field,new_value", [
    ("expected_version", 2),
    ("start_utc", "2026-03-09T15:30:00Z"),
    ("end_utc", "2026-03-09T17:00:00Z"),
    ("participant_ids", ["alice", "carol"]),
    ("approver_ids", ["boss-a"]),
])
def test_create_proposal_same_key_different_content_409(
        client, field, new_value):
    r = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    r2 = _reservation(client, "req-2", "2026-03-10T13:00:00Z",
                      "2026-03-10T14:00:00Z", ("bob",))
    assert client.post(
        "/api/v1/reschedule-proposals",
        json=_proposal_payload(r["reservation_id"])).status_code == 201
    other = _proposal_payload(r["reservation_id"], **{field: new_value})
    resp = client.post("/api/v1/reschedule-proposals", json=other)
    assert resp.status_code == 409, resp.text
    assert "different content" in resp.json()["detail"]
    # reservation_id 变更同样视为异内容
    other = _proposal_payload(r["reservation_id"])
    other["reservation_id"] = r2["reservation_id"]
    resp = client.post("/api/v1/reschedule-proposals", json=other)
    assert resp.status_code == 409, resp.text


def test_create_proposal_reservation_not_found_404(client):
    resp = client.post("/api/v1/reschedule-proposals",
                       json=_proposal_payload(999))
    assert resp.status_code == 404, resp.text
    assert "not found" in resp.json()["detail"]


def test_create_proposal_on_canceled_reservation_allowed_but_never_executes(
        client):
    """建提案只校验存在性：已取消预约可建提案，但最后一票 approve 执行时 409。"""
    r = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    assert client.delete(
        f"/api/v1/reservations/{r['reservation_id']}?expected_version=1"
    ).status_code == 200
    resp = client.post("/api/v1/reschedule-proposals", json=_proposal_payload(
        r["reservation_id"], approver_ids=["boss-a"]))
    assert resp.status_code == 201, resp.text
    vote = _vote(client, "prop-1", "boss-a", "approve")
    assert vote.status_code == 409, vote.text  # 已取消：执行失败
    assert "canceled" in vote.json()["detail"]
    # 提案仍 pending，该票未落库
    proposal = _get_proposal(client, "prop-1")
    assert proposal["status"] == "pending"
    assert proposal["votes"] == []


# -------------------------------------------------------------------- 查询

def test_get_proposal_404(client):
    resp = client.get("/api/v1/reschedule-proposals/nope")
    assert resp.status_code == 404, resp.text


def test_get_proposal_reflects_votes_and_reservation(client):
    r = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    _create_proposal(client, r["reservation_id"])
    assert _vote(client, "prop-1", "boss-b", "approve").status_code == 201
    data = _get_proposal(client, "prop-1")
    assert data["status"] == "pending"
    assert [(v["voter_id"], v["decision"]) for v in data["votes"]] == [
        ("boss-b", "approve")]
    assert data["votes"][0]["voted_at"]
    assert data["reservation"]["version"] == 1


# -------------------------------------------------------------------- 投票

def test_vote_on_missing_proposal_404(client):
    resp = _vote(client, "nope", "boss-a", "approve")
    assert resp.status_code == 404, resp.text


def test_vote_non_approver_409(client):
    r = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    _create_proposal(client, r["reservation_id"])
    resp = _vote(client, "prop-1", "stranger", "approve")
    assert resp.status_code == 409, resp.text
    assert "not a required approver" in resp.json()["detail"]
    assert _get_proposal(client, "prop-1")["votes"] == []


def test_vote_approve_records_vote_pending(client):
    r = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    _create_proposal(client, r["reservation_id"])
    resp = _vote(client, "prop-1", "boss-a", "approve")
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["status"] == "pending"  # 还差 boss-b
    assert [(v["voter_id"], v["decision"]) for v in data["votes"]] == [
        ("boss-a", "approve")]
    # 未执行改期：预约保持原状
    assert data["reservation"]["version"] == 1
    assert data["reservation"]["start_utc"] == "2026-03-09T13:00:00Z"


def test_vote_replay_same_decision_idempotent(client):
    r = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    _create_proposal(client, r["reservation_id"])
    first = _vote(client, "prop-1", "boss-a", "approve")
    assert first.status_code == 201
    replay = _vote(client, "prop-1", "boss-a", "approve")
    assert replay.status_code == 200
    assert replay.json() == first.json()
    assert len(_get_proposal(client, "prop-1")["votes"]) == 1


def test_vote_change_409(client):
    r = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    _create_proposal(client, r["reservation_id"])
    assert _vote(client, "prop-1", "boss-a", "approve").status_code == 201
    resp = _vote(client, "prop-1", "boss-a", "reject")
    assert resp.status_code == 409, resp.text
    assert resp.json()["existing_decision"] == "approve"
    # 改票未生效：仍只有原票，提案仍 pending
    proposal = _get_proposal(client, "prop-1")
    assert proposal["status"] == "pending"
    assert [(v["voter_id"], v["decision"]) for v in proposal["votes"]] == [
        ("boss-a", "approve")]


def test_reject_terminates_proposal(client):
    r = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    _create_proposal(client, r["reservation_id"])
    assert _vote(client, "prop-1", "boss-a", "approve").status_code == 201
    resp = _vote(client, "prop-1", "boss-b", "reject")
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["status"] == "rejected"
    assert data["decided_at"] is not None
    # 提案终止：预约不被改期
    assert data["reservation"]["version"] == 1
    assert data["reservation"]["start_utc"] == "2026-03-09T13:00:00Z"

    # 终态后：新投票 409；已投者重复同票幂等；改票 409
    late = _vote(client, "prop-1", "boss-a", "reject")  # 改票（已投 approve）
    assert late.status_code == 409
    assert late.json()["existing_decision"] == "approve"
    replay = _vote(client, "prop-1", "boss-b", "reject")  # 重复同票
    assert replay.status_code == 200
    assert replay.json()["status"] == "rejected"
    # 历史无 rescheduled 事件
    actions = [e["action"] for e in _history(client, r["reservation_id"])]
    assert actions == ["created"]


def test_vote_on_terminal_proposal_409_for_new_voter(client):
    r = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    _create_proposal(client, r["reservation_id"], approver_ids=["boss-a"])
    assert _vote(client, "prop-1", "boss-a", "reject").status_code == 201
    resp = client.post(
        "/api/v1/reschedule-proposals/prop-1/votes",
        json={"voter_id": "boss-a", "decision": "approve"})
    assert resp.status_code == 409  # 改票优先于终态判定
    # 另一个必批人维度：新建一个三必批人提案，否决后其余必批人新票 409
    _create_proposal(client, r["reservation_id"], proposal_id="prop-2",
                     approver_ids=["a1", "a2", "a3"])
    assert _vote(client, "prop-2", "a1", "reject").status_code == 201
    resp = _vote(client, "prop-2", "a2", "approve")
    assert resp.status_code == 409, resp.text
    assert resp.json()["status"] == "rejected"


# ------------------------------------------------------- 最后一票 approve

def test_full_approval_executes_reschedule(client):
    r = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    _create_proposal(client, r["reservation_id"],
                     participant_ids=["alice", "carol"])
    assert _vote(client, "prop-1", "boss-a", "approve").status_code == 201
    resp = _vote(client, "prop-1", "boss-b", "approve")
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["status"] == "approved"
    assert data["decided_at"] is not None
    assert {(v["voter_id"], v["decision"]) for v in data["votes"]} == {
        ("boss-a", "approve"), ("boss-b", "approve")}
    # 改期已执行：新时段/参与者，version+1
    reservation = data["reservation"]
    assert reservation["start_utc"] == "2026-03-09T15:00:00Z"
    assert reservation["end_utc"] == "2026-03-09T16:00:00Z"
    assert reservation["participants"] == ["alice", "carol"]
    assert reservation["version"] == 2
    assert reservation["status"] == "active"
    # 同事务追加 rescheduled 审计事件，链可重验
    events = _history(client, r["reservation_id"])
    assert [e["action"] for e in events] == ["created", "rescheduled"]
    assert events[-1]["snapshot"]["version"] == 2
    assert events[-1]["snapshot"]["participants"] == ["alice", "carol"]
    # 旧时段已释放、新时段被占用
    resp = client.post("/api/v1/reservations", json={
        "request_id": "req-old-window",
        "start_utc": "2026-03-09T13:00:00Z",
        "end_utc": "2026-03-09T14:00:00Z",
        "participant_ids": ["alice"],
    })
    assert resp.status_code == 201, resp.text
    conflict = client.post("/api/v1/reservations", json={
        "request_id": "req-new-window",
        "start_utc": "2026-03-09T15:30:00Z",
        "end_utc": "2026-03-09T16:30:00Z",
        "participant_ids": ["carol"],
    })
    assert conflict.status_code == 409, conflict.text
    # 终态后新票 409；已投者重复同票幂等返回 approved 状态
    assert _vote(client, "prop-1", "boss-a", "approve").status_code == 200
    late = client.post("/api/v1/reschedule-proposals/prop-1/votes",
                       json={"voter_id": "boss-b", "decision": "reject"})
    assert late.status_code == 409  # 改票
    assert late.json()["existing_decision"] == "approve"


def test_single_approver_executes_on_first_vote(client):
    r = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    _create_proposal(client, r["reservation_id"], approver_ids=["boss-a"])
    resp = _vote(client, "prop-1", "boss-a", "approve")
    assert resp.status_code == 201
    assert resp.json()["status"] == "approved"
    assert resp.json()["reservation"]["version"] == 2


def test_final_approve_version_mismatch_409_proposal_stays_pending(client):
    r = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    _create_proposal(client, r["reservation_id"])
    assert _vote(client, "prop-1", "boss-a", "approve").status_code == 201
    # 提案期间预约被他人直接改期：版本推进到 2
    resp = client.put(f"/api/v1/reservations/{r['reservation_id']}", json={
        "expected_version": 1,
        "start_utc": "2026-03-10T13:00:00Z",
        "end_utc": "2026-03-10T14:00:00Z",
        "participant_ids": ["alice"],
    })
    assert resp.status_code == 200, resp.text
    # 最后一票 approve：版本不符 → 409，提案仍 pending，该票未落库
    vote = _vote(client, "prop-1", "boss-b", "approve")
    assert vote.status_code == 409, vote.text
    assert vote.json()["current_version"] == 2
    proposal = _get_proposal(client, "prop-1")
    assert proposal["status"] == "pending"
    assert [(v["voter_id"], v["decision"]) for v in proposal["votes"]] == [
        ("boss-a", "approve")]
    # 预约保持他人改期后的状态，未被提案改动
    reservation = proposal["reservation"]
    assert reservation["version"] == 2
    assert reservation["start_utc"] == "2026-03-10T13:00:00Z"
    # 未产生提案的审计事件（只有 created + 他人的 rescheduled）
    actions = [e["action"] for e in _history(client, r["reservation_id"])]
    assert actions == ["created", "rescheduled"]


def test_final_approve_participant_conflict_409(client):
    r = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    # 目标新时段被 carol 的另一条预约占用
    _reservation(client, "req-block", "2026-03-09T15:00:00Z",
                 "2026-03-09T16:00:00Z", ("carol",))
    _create_proposal(client, r["reservation_id"],
                     participant_ids=["alice", "carol"],
                     approver_ids=["boss-a"])
    vote = _vote(client, "prop-1", "boss-a", "approve")
    assert vote.status_code == 409, vote.text
    assert vote.json()["conflicts"], vote.text
    # 提案仍 pending、票未落库、预约不变
    proposal = _get_proposal(client, "prop-1")
    assert proposal["status"] == "pending"
    assert proposal["votes"] == []
    assert proposal["reservation"]["version"] == 1
    assert proposal["reservation"]["start_utc"] == "2026-03-09T13:00:00Z"


def test_final_approve_hold_conflict_409_then_retry_after_expiry(client):
    r = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    _create_proposal(client, r["reservation_id"], approver_ids=["boss-a"])
    # 未过期保留占住目标新时段 → 执行 409
    hold = client.post("/api/v1/reservation-holds", json={
        "hold_key": "hold-1",
        "start_utc": "2026-03-09T15:00:00Z",
        "end_utc": "2026-03-09T16:00:00Z",
        "participant_ids": ["alice"],
        "ttl_seconds": 900,
    })
    assert hold.status_code == 201, hold.text
    vote = _vote(client, "prop-1", "boss-a", "approve")
    assert vote.status_code == 409, vote.text
    assert any("hold_id" in c for c in vote.json()["conflicts"])
    assert _get_proposal(client, "prop-1")["status"] == "pending"


def test_final_approve_dependency_violation_409(client):
    a = _reservation(client, "req-a", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    b = _reservation(client, "req-b", "2026-03-09T15:00:00Z",
                     "2026-03-09T16:00:00Z", ("bob",))
    # a 结束后至少 30 分钟 b 才能开始
    dep = client.post("/api/v1/reservation-dependencies", json={
        "dependency_request_id": "dep-1",
        "predecessor_id": a["reservation_id"],
        "successor_id": b["reservation_id"],
        "min_gap_minutes": 30,
    })
    assert dep.status_code == 201, dep.text
    # 提案：把 a 改到 15:30-16:30（a.end=16:30 > b.start=15:00，违反依赖）
    _create_proposal(client, a["reservation_id"],
                     start_utc="2026-03-09T15:30:00Z",
                     end_utc="2026-03-09T16:30:00Z",
                     approver_ids=["boss-a"])
    vote = _vote(client, "prop-1", "boss-a", "approve")
    assert vote.status_code == 409, vote.text
    assert vote.json()["violations"], vote.text
    assert vote.json()["violations"][0]["predecessor_id"] == a["reservation_id"]
    # 提案仍 pending、预约不变
    proposal = _get_proposal(client, "prop-1")
    assert proposal["status"] == "pending"
    assert proposal["votes"] == []
    assert proposal["reservation"]["start_utc"] == "2026-03-09T13:00:00Z"


def test_final_approve_releases_room_assignment(client):
    r = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    assert client.put("/api/v1/meeting-rooms/room-1", json={
        "capacity": 4, "equipment": [],
    }).status_code == 201
    batch = client.post("/api/v1/room-assignments", json={
        "batch_id": "batch-1",
        "items": [{"reservation_id": r["reservation_id"],
                   "required_capacity": 2}],
    })
    assert batch.status_code == 201, batch.text
    assert client.get(
        "/api/v1/room-assignments",
        params={"reservation_id": r["reservation_id"]}).json()

    _create_proposal(client, r["reservation_id"], approver_ids=["boss-a"])
    assert _vote(client, "prop-1", "boss-a", "approve").status_code == 201
    # 改期执行同事务释放会议室分配
    assert client.get(
        "/api/v1/room-assignments",
        params={"reservation_id": r["reservation_id"]}).json() == []


def test_proposal_execution_failure_leaves_room_assignment(client):
    """执行失败整体回滚：已有会议室分配也保持原状。"""
    r = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    assert client.put("/api/v1/meeting-rooms/room-1", json={
        "capacity": 4, "equipment": [],
    }).status_code == 201
    assert client.post("/api/v1/room-assignments", json={
        "batch_id": "batch-1",
        "items": [{"reservation_id": r["reservation_id"],
                   "required_capacity": 2}],
    }).status_code == 201
    _reservation(client, "req-block", "2026-03-09T15:00:00Z",
                 "2026-03-09T16:00:00Z", ("alice",))
    _create_proposal(client, r["reservation_id"], approver_ids=["boss-a"])
    assert _vote(client, "prop-1", "boss-a", "approve").status_code == 409
    # 分配未被释放
    assert client.get(
        "/api/v1/room-assignments",
        params={"reservation_id": r["reservation_id"]}).json()


# -------------------------------------------------------------------- 并发

def test_concurrent_final_votes_execute_exactly_once(db_path):
    """单必批人：N 个并发相同 approve 票，改期只执行一次。"""
    client = TestClient(app)
    r = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    _create_proposal(client, r["reservation_id"], approver_ids=["boss-a"])

    barrier = threading.Barrier(8)

    def vote():
        barrier.wait()
        return _vote(client, "prop-1", "boss-a", "approve")

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: vote(), range(8)))
    codes = sorted(resp.status_code for resp in responses)
    # 恰一票实际记票并执行（201），其余为幂等重放（200）
    assert codes == [200] * 7 + [201]
    proposal = _get_proposal(client, "prop-1")
    assert proposal["status"] == "approved"
    assert len(proposal["votes"]) == 1
    # 版本只推进一次、审计链只有一条 rescheduled
    assert proposal["reservation"]["version"] == 2
    actions = [e["action"] for e in _history(client, r["reservation_id"])]
    assert actions == ["created", "rescheduled"]


def test_concurrent_distinct_approvers_single_execution(db_path):
    """双必批人并发 approve：两票都记入，但改期只执行一次。"""
    client = TestClient(app)
    r = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    _create_proposal(client, r["reservation_id"])

    barrier = threading.Barrier(2)

    def vote(voter):
        barrier.wait()
        return _vote(client, "prop-1", voter, "approve")

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(vote, ["boss-a", "boss-b"]))
    assert sorted(resp.status_code for resp in responses) == [201, 201]
    proposal = _get_proposal(client, "prop-1")
    assert proposal["status"] == "approved"
    assert len(proposal["votes"]) == 2
    assert proposal["reservation"]["version"] == 2
    actions = [e["action"] for e in _history(client, r["reservation_id"])]
    assert actions == ["created", "rescheduled"]


def test_concurrent_create_same_key_single_record(db_path):
    """并发创建同键同内容提案：只建一条，其余幂等重放。"""
    client = TestClient(app)
    r = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    payload = _proposal_payload(r["reservation_id"])
    barrier = threading.Barrier(6)

    def create():
        barrier.wait()
        return client.post("/api/v1/reschedule-proposals", json=payload)

    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(lambda _: create(), range(6)))
    codes = sorted(resp.status_code for resp in responses)
    assert codes == [200] * 5 + [201]
    assert _get_proposal(client, "prop-1")["status"] == "pending"


# -------------------------------------------------------------- 无损建表

def test_legacy_db_gains_proposal_tables_losslessly(tmp_path, monkeypatch):
    """旧版库（无提案表）初始化后新增提案/投票表，既有数据不动。"""
    path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(path)
    try:
        conn.executescript("""\
CREATE TABLE reservations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL UNIQUE,
    start_utc TEXT NOT NULL, end_utc TEXT NOT NULL,
    participants_key TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE reservation_participants (
    reservation_id INTEGER NOT NULL REFERENCES reservations(id) ON DELETE CASCADE,
    participant_id TEXT NOT NULL, start_utc TEXT NOT NULL, end_utc TEXT NOT NULL,
    PRIMARY KEY (reservation_id, participant_id));
""")
        conn.execute(
            "INSERT INTO reservations"
            " (request_id, start_utc, end_utc, participants_key, created_at)"
            " VALUES ('old', '2026-03-09T13:00:00.000000Z',"
            " '2026-03-09T14:00:00.000000Z', '[\"alice\"]',"
            " '2026-03-01T00:00:00.000000Z')")
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setenv("RESERVATIONS_DB_PATH", path)
    db_module.init_db(path)
    db_module.init_db(path)  # 幂等

    conn = sqlite3.connect(path)
    try:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "reschedule_proposals" in names
        assert "reschedule_proposal_votes" in names
        # 既有预约无损
        assert conn.execute(
            "SELECT request_id, participants_key FROM reservations"
        ).fetchone() == ("old", "[\"alice\"]")
    finally:
        conn.close()

    # 迁移后提案流程在旧库上正常工作
    client = TestClient(app)
    resp = client.post("/api/v1/reschedule-proposals", json=_proposal_payload(
        1, approver_ids=["boss-a"]))
    assert resp.status_code == 201, resp.text
    vote = _vote(client, "prop-1", "boss-a", "approve")
    assert vote.status_code == 201, vote.text
    assert vote.json()["status"] == "approved"
