"""预约改期提案（reschedule proposals）测试。

覆盖：创建提案（201/幂等重放/同键异内容 409/预约缺失 404/校验 422）、
投票（非必批人 409/重复同票幂等/改票 409/已终止 409/任一 reject 终止）、
最后一票 approve 同事务改期（版本推进、审计事件、会议室分配释放）、
最后一票失败（版本漂移/已取消/参与者冲突/保留冲突/依赖违反 → 409，
提案仍 pending、预约不变、该票不落库，条件满足后重试可成功）、
并发最后投票只提交一次、旧版库无损建新表。
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app import db as db_module
from app import proposals as proposal_core
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


def _reservation(client, request_id="req-1", start="2026-03-09T13:00:00Z",
                 end="2026-03-09T14:00:00Z", participants=("alice",)):
    resp = client.post("/api/v1/reservations", json={
        "request_id": request_id,
        "start_utc": start,
        "end_utc": end,
        "participant_ids": list(participants),
    })
    assert resp.status_code == 201, resp.text
    return resp.json()


def _proposal_payload(rid=None, **overrides):
    payload = {
        "proposal_id": "prop-1",
        "reservation_id": rid,
        "expected_version": 1,
        "start_utc": "2026-03-09T15:00:00Z",
        "end_utc": "2026-03-09T16:00:00Z",
        "participant_ids": ["alice"],
        "approver_ids": ["boss1", "boss2"],
    }
    payload.update(overrides)
    return payload


def _create_proposal(client, reservation_id, **overrides):
    resp = client.post("/api/v1/reschedule-proposals",
                       json=_proposal_payload(reservation_id, **overrides))
    assert resp.status_code == 201, resp.text
    return resp.json()


def _vote(client, proposal_id, approver_id, decision):
    return client.post(f"/api/v1/reschedule-proposals/{proposal_id}/votes",
                       json={"approver_id": approver_id, "decision": decision})


def _get_proposal(client, proposal_id):
    resp = client.get(f"/api/v1/reschedule-proposals/{proposal_id}")
    assert resp.status_code == 200, resp.text
    return resp.json()


def _history(client, reservation_id):
    resp = client.get(f"/api/v1/reservations/{reservation_id}/history")
    assert resp.status_code == 200, resp.text
    return resp.json()["events"]


# ------------------------------------------------------------------ 创建提案

def test_create_proposal_returns_201(client):
    r = _reservation(client)
    resp = client.post("/api/v1/reschedule-proposals",
                       json=_proposal_payload(r["reservation_id"]))
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["proposal_id"] == "prop-1"
    assert data["reservation_id"] == r["reservation_id"]
    assert data["expected_version"] == 1
    assert data["start_utc"] == "2026-03-09T15:00:00Z"
    assert data["end_utc"] == "2026-03-09T16:00:00Z"
    assert data["participants"] == ["alice"]
    assert data["approvers"] == ["boss1", "boss2"]
    assert data["status"] == "pending"
    assert data["votes"] == []
    assert data["resolved_at"] is None
    assert data["created_at"]
    # 响应内嵌目标预约当前状态
    assert data["reservation"]["reservation_id"] == r["reservation_id"]
    assert data["reservation"]["version"] == 1
    assert data["reservation"]["status"] == "active"


def test_create_proposal_normalizes_participants_and_approvers(client):
    r = _reservation(client)
    resp = client.post("/api/v1/reschedule-proposals",
                       json=_proposal_payload(
                           r["reservation_id"],
                           participant_ids=["carol", "alice", "carol"],
                           approver_ids=["boss2", "boss1", "boss2"]))
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["participants"] == ["alice", "carol"]
    assert data["approvers"] == ["boss1", "boss2"]


def test_create_proposal_idempotent_replay(client):
    r = _reservation(client)
    payload = _proposal_payload(r["reservation_id"])
    first = client.post("/api/v1/reschedule-proposals", json=payload)
    assert first.status_code == 201
    replay = client.post("/api/v1/reschedule-proposals", json=payload)
    assert replay.status_code == 200
    assert replay.json() == first.json()  # 返回原提案，不新增记录
    # 规范化等价（参与者/必批人顺序不同、含重复）也按同内容幂等
    equivalent = dict(payload, participant_ids=["alice", "alice"],
                      approver_ids=["boss2", "boss1", "boss2"])
    replay2 = client.post("/api/v1/reschedule-proposals", json=equivalent)
    assert replay2.status_code == 200
    assert replay2.json() == first.json()


@pytest.mark.parametrize("field,value", [
    ("expected_version", 2),
    ("start_utc", "2026-03-09T15:30:00Z"),
    ("end_utc", "2026-03-09T16:30:00Z"),
    ("participant_ids", ["alice", "carol"]),
    ("approver_ids", ["boss1"]),
])
def test_create_proposal_same_key_different_content_409(client, field, value):
    r = _reservation(client)
    other_res = _reservation(client, "req-2", "2026-03-10T13:00:00Z",
                             "2026-03-10T14:00:00Z", ("bob",))
    payload = _proposal_payload(r["reservation_id"])
    assert client.post("/api/v1/reschedule-proposals",
                       json=payload).status_code == 201
    other = dict(payload)
    other[field] = value
    resp = client.post("/api/v1/reschedule-proposals", json=other)
    assert resp.status_code == 409
    data = resp.json()
    assert data["existing_proposal_id"] == "prop-1"
    assert "different content" in data["detail"]
    # 目标预约不同同样是异内容
    other = dict(payload, reservation_id=other_res["reservation_id"])
    resp = client.post("/api/v1/reschedule-proposals", json=other)
    assert resp.status_code == 409


def test_create_proposal_missing_reservation_404(client):
    resp = client.post("/api/v1/reschedule-proposals",
                       json=_proposal_payload(999))
    assert resp.status_code == 404
    assert resp.json()["detail"] == "reservation #999 not found"


def test_create_proposal_defers_version_and_status_checks(client):
    """创建只校验预约存在：版本漂移/已取消的校验推迟到最后一票。"""
    r = _reservation(client)
    # expected_version 与当前版本不一致：创建仍 201（最后一票时才 409）
    resp = client.post("/api/v1/reschedule-proposals",
                       json=_proposal_payload(r["reservation_id"],
                                              expected_version=7))
    assert resp.status_code == 201, resp.text
    # 已取消预约也可创建提案（同样推迟到最后一票）
    resp = client.delete(
        f"/api/v1/reservations/{r['reservation_id']}?expected_version=1")
    assert resp.status_code == 200
    resp = client.post("/api/v1/reschedule-proposals",
                       json=_proposal_payload(r["reservation_id"],
                                              proposal_id="prop-2",
                                              expected_version=2))
    assert resp.status_code == 201, resp.text


@pytest.mark.parametrize("override", [
    {"approver_ids": []},
    {"approver_ids": ["boss1", ""]},
    {"expected_version": 0},
    {"reservation_id": 0},
    {"start_utc": "2026-03-09T16:00:00Z", "end_utc": "2026-03-09T15:00:00Z"},
    {"participant_ids": []},
])
def test_create_proposal_validation_422(client, override):
    r = _reservation(client)
    resp = client.post("/api/v1/reschedule-proposals",
                       json=_proposal_payload(r["reservation_id"], **override))
    assert resp.status_code == 422


def test_create_proposal_core_empty_approvers_defense(db_path):
    """绕过 API 层直接调核心：空必批人列表被拒绝。"""
    from app import reservations as reservation_core

    reservation, _ = reservation_core.create_reservation(
        request_id="req-1", start_utc=_utc(2026, 3, 9, 13),
        end_utc=_utc(2026, 3, 9, 14), participant_ids=["alice"],
        db_path=db_path)
    with pytest.raises(ValueError):
        proposal_core.create_proposal(
            proposal_id="prop-1", reservation_id=reservation.id,
            expected_version=1, start_utc=_utc(2026, 3, 9, 15),
            end_utc=_utc(2026, 3, 9, 16), participant_ids=["alice"],
            approver_ids=[], db_path=db_path)


# ------------------------------------------------------------------ 投票

def test_vote_records_approve_and_keeps_pending(client):
    r = _reservation(client)
    _create_proposal(client, r["reservation_id"])
    resp = _vote(client, "prop-1", "boss1", "approve")
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["status"] == "pending"  # 还差 boss2
    vote, = data["votes"]
    assert vote["approver_id"] == "boss1"
    assert vote["decision"] == "approve"
    assert vote["voted_at"]
    # 预约未被改动
    assert data["reservation"]["version"] == 1
    assert data["reservation"]["start_utc"] == "2026-03-09T13:00:00Z"


def test_vote_same_vote_idempotent_replay(client):
    r = _reservation(client)
    _create_proposal(client, r["reservation_id"])
    first = _vote(client, "prop-1", "boss1", "approve")
    assert first.status_code == 201
    replay = _vote(client, "prop-1", "boss1", "approve")
    assert replay.status_code == 200
    assert replay.json() == first.json()
    assert len(replay.json()["votes"]) == 1  # 不重复记录


def test_vote_change_rejected_409(client):
    r = _reservation(client)
    _create_proposal(client, r["reservation_id"])
    assert _vote(client, "prop-1", "boss1", "approve").status_code == 201
    resp = _vote(client, "prop-1", "boss1", "reject")
    assert resp.status_code == 409
    data = resp.json()
    assert data["existing_decision"] == "approve"
    assert "already voted" in data["detail"]
    # 改票被拒绝不终止提案：仍 pending，boss2 可继续投票
    assert _get_proposal(client, "prop-1")["status"] == "pending"
    assert _vote(client, "prop-1", "boss2", "approve").status_code == 201


def test_vote_by_non_approver_409(client):
    r = _reservation(client)
    _create_proposal(client, r["reservation_id"])
    resp = _vote(client, "prop-1", "stranger", "approve")
    assert resp.status_code == 409
    assert "not a required approver" in resp.json()["detail"]
    assert _get_proposal(client, "prop-1")["votes"] == []


def test_vote_on_missing_proposal_404(client):
    resp = _vote(client, "prop-nope", "boss1", "approve")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "reschedule proposal 'prop-nope' not found"


def test_get_missing_proposal_404(client):
    resp = client.get("/api/v1/reschedule-proposals/prop-nope")
    assert resp.status_code == 404


def test_vote_invalid_decision_422(client):
    r = _reservation(client)
    _create_proposal(client, r["reservation_id"])
    resp = _vote(client, "prop-1", "boss1", "abstain")
    assert resp.status_code == 422


def test_reject_terminates_proposal(client):
    r = _reservation(client)
    _create_proposal(client, r["reservation_id"])
    assert _vote(client, "prop-1", "boss1", "approve").status_code == 201
    resp = _vote(client, "prop-1", "boss2", "reject")
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["status"] == "rejected"
    assert data["resolved_at"] is not None
    assert len(data["votes"]) == 2
    # 预约保持原状（版本、时段、审计事件均不变）
    assert data["reservation"]["version"] == 1
    assert data["reservation"]["start_utc"] == "2026-03-09T13:00:00Z"
    events = _history(client, r["reservation_id"])
    assert [e["action"] for e in events] == ["created"]
    # 已终止提案不再接受新投票
    resp = _vote(client, "prop-1", "boss3", "approve")
    assert resp.status_code == 409  # boss3 本就不是必批人
    # 重复同一否决票：幂等返回当前（已终止）提案
    replay = _vote(client, "prop-1", "boss2", "reject")
    assert replay.status_code == 200
    assert replay.json()["status"] == "rejected"
    # 已投票人改票仍 409
    assert _vote(client, "prop-1", "boss2", "approve").status_code == 409


def test_vote_on_approved_proposal_closed_409(client):
    r = _reservation(client)
    _create_proposal(client, r["reservation_id"], approver_ids=["boss1", "boss2"])
    assert _vote(client, "prop-1", "boss1", "approve").status_code == 201
    assert _vote(client, "prop-1", "boss2", "approve").status_code == 201
    assert _get_proposal(client, "prop-1")["status"] == "approved"
    # 已通过的提案不再接受新投票（重复同票仍幂等 200）
    replay = _vote(client, "prop-1", "boss1", "approve")
    assert replay.status_code == 200
    assert replay.json()["status"] == "approved"


# -------------------------------------------------------- 最后一票 approve

def test_final_approve_executes_reschedule(client):
    r = _reservation(client)
    _create_proposal(client, r["reservation_id"],
                     participant_ids=["alice", "carol"])
    assert _vote(client, "prop-1", "boss1", "approve").status_code == 201
    resp = _vote(client, "prop-1", "boss2", "approve")
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["status"] == "approved"
    assert data["resolved_at"] is not None
    assert {v["approver_id"] for v in data["votes"]} == {"boss1", "boss2"}
    # 预约已在同事务内改期：新时段/参与者，版本 +1
    reservation = data["reservation"]
    assert reservation["version"] == 2
    assert reservation["start_utc"] == "2026-03-09T15:00:00Z"
    assert reservation["end_utc"] == "2026-03-09T16:00:00Z"
    assert reservation["participants"] == ["alice", "carol"]
    assert reservation["request_id"] == "req-1"
    # 同事务追加 rescheduled 审计事件，链可重验
    events = _history(client, r["reservation_id"])
    assert [e["action"] for e in events] == ["created", "rescheduled"]
    assert events[-1]["snapshot"]["version"] == 2
    assert events[-1]["snapshot"]["start_utc"] == "2026-03-09T15:00:00Z"
    # 旧时段已释放、新时段被占用
    ok = client.post("/api/v1/reservations", json={
        "request_id": "req-old-window",
        "start_utc": "2026-03-09T13:00:00Z",
        "end_utc": "2026-03-09T14:00:00Z",
        "participant_ids": ["alice"],
    })
    assert ok.status_code == 201, ok.text
    conflict = client.post("/api/v1/reservations", json={
        "request_id": "req-new-window",
        "start_utc": "2026-03-09T15:30:00Z",
        "end_utc": "2026-03-09T16:30:00Z",
        "participant_ids": ["carol"],
    })
    assert conflict.status_code == 409


def test_final_approve_single_approver(client):
    r = _reservation(client)
    _create_proposal(client, r["reservation_id"], approver_ids=["boss1"])
    resp = _vote(client, "prop-1", "boss1", "approve")
    assert resp.status_code == 201
    assert resp.json()["status"] == "approved"
    assert resp.json()["reservation"]["version"] == 2


def test_final_approve_releases_room_assignment(client):
    r = _reservation(client)
    room = client.put("/api/v1/meeting-rooms/room-a",
                      json={"capacity": 4, "equipment": []})
    assert room.status_code == 201
    batch = client.post("/api/v1/room-assignments", json={
        "batch_id": "batch-1",
        "items": [{"reservation_id": r["reservation_id"],
                   "required_capacity": 2}],
    })
    assert batch.status_code == 201, batch.text
    assert client.get("/api/v1/room-assignments").json() != []

    _create_proposal(client, r["reservation_id"], approver_ids=["boss1"])
    resp = _vote(client, "prop-1", "boss1", "approve")
    assert resp.status_code == 201, resp.text
    assert resp.json()["status"] == "approved"
    # 分配与求解时的时段绑定：提案改期通过即同事务释放
    assert client.get("/api/v1/room-assignments").json() == []


def test_final_approve_version_mismatch_409(client):
    """投票期间预约被他人改期：最后一票 409，提案仍 pending、预约不变。"""
    r = _reservation(client)
    _create_proposal(client, r["reservation_id"], approver_ids=["boss1"])
    # 提案外有人把预约改期（version 1 -> 2）
    moved = client.put(f"/api/v1/reservations/{r['reservation_id']}", json={
        "expected_version": 1,
        "start_utc": "2026-03-09T17:00:00Z",
        "end_utc": "2026-03-09T18:00:00Z",
        "participant_ids": ["alice"],
    })
    assert moved.status_code == 200, moved.text
    resp = _vote(client, "prop-1", "boss1", "approve")
    assert resp.status_code == 409
    assert resp.json()["current_version"] == 2
    # 提案仍 pending、该票未落库、预约保持他人改期后的状态
    proposal = _get_proposal(client, "prop-1")
    assert proposal["status"] == "pending"
    assert proposal["votes"] == []
    assert proposal["reservation"]["version"] == 2
    assert proposal["reservation"]["start_utc"] == "2026-03-09T17:00:00Z"
    # 不产生新的审计事件
    events = _history(client, r["reservation_id"])
    assert [e["action"] for e in events] == ["created", "rescheduled"]


def test_final_approve_canceled_reservation_409(client):
    r = _reservation(client)
    _create_proposal(client, r["reservation_id"], approver_ids=["boss1"])
    resp = client.delete(
        f"/api/v1/reservations/{r['reservation_id']}?expected_version=1")
    assert resp.status_code == 200
    resp = _vote(client, "prop-1", "boss1", "approve")
    assert resp.status_code == 409
    assert "canceled" in resp.json()["detail"]
    proposal = _get_proposal(client, "prop-1")
    assert proposal["status"] == "pending"
    assert proposal["votes"] == []
    # 取消是终态：提案改期不会使其复活
    events = _history(client, r["reservation_id"])
    assert [e["action"] for e in events] == ["created", "canceled"]


def test_final_approve_participant_conflict_409_then_retry(client):
    """最后一票撞上冲突：409 且整体回滚；障碍消除后重试同一票成功。"""
    r = _reservation(client)
    _create_proposal(client, r["reservation_id"], approver_ids=["boss1"])
    # 提案创建后，目标新时段被他人（共享参与者 alice）占用
    blocker = _reservation(client, "req-blocker", "2026-03-09T15:00:00Z",
                           "2026-03-09T16:00:00Z", ("alice",))
    resp = _vote(client, "prop-1", "boss1", "approve")
    assert resp.status_code == 409
    assert resp.json()["conflicts"] == [
        {"reservation_id": blocker["reservation_id"], "participant_id": "alice"}
    ]
    # 提案仍 pending、该票未落库、预约不变、无新审计事件
    proposal = _get_proposal(client, "prop-1")
    assert proposal["status"] == "pending"
    assert proposal["votes"] == []
    assert proposal["reservation"]["version"] == 1
    events = _history(client, r["reservation_id"])
    assert [e["action"] for e in events] == ["created"]
    # 障碍消除后重试同一票：成功改期
    resp = client.delete(
        f"/api/v1/reservations/{blocker['reservation_id']}?expected_version=1")
    assert resp.status_code == 200
    retry = _vote(client, "prop-1", "boss1", "approve")
    assert retry.status_code == 201, retry.text
    assert retry.json()["status"] == "approved"
    assert retry.json()["reservation"]["version"] == 2


def test_final_approve_hold_conflict_409(client):
    """最后一票撞上未过期保留：409，提案仍 pending。"""
    r = _reservation(client)
    _create_proposal(client, r["reservation_id"], approver_ids=["boss1"])
    hold = client.post("/api/v1/reservation-holds", json={
        "hold_key": "hold-1",
        "start_utc": "2026-03-09T15:00:00Z",
        "end_utc": "2026-03-09T16:00:00Z",
        "participant_ids": ["alice"],
        "ttl_seconds": 900,
    })
    assert hold.status_code == 201, hold.text
    resp = _vote(client, "prop-1", "boss1", "approve")
    assert resp.status_code == 409
    assert resp.json()["conflicts"][0]["hold_id"] == hold.json()["hold_id"]
    assert _get_proposal(client, "prop-1")["status"] == "pending"


def test_final_approve_dependency_violation_409(client):
    """最后一票时重验可达依赖：违反则 409，提案仍 pending、预约不变。"""
    a = _reservation(client, "req-a", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    b = _reservation(client, "req-b", "2026-03-09T15:00:00Z",
                     "2026-03-09T16:00:00Z", ("bob",))
    dep = client.post("/api/v1/reservation-dependencies", json={
        "dependency_request_id": "dep-1",
        "predecessor_id": a["reservation_id"],
        "successor_id": b["reservation_id"],
        "min_gap_minutes": 30,
    })
    assert dep.status_code == 201, dep.text
    # 提案把 A 改到 13:00-14:31：14:31 + 30m = 15:01 > B.start 15:00 → 违反
    _create_proposal(client, a["reservation_id"], approver_ids=["boss1"],
                     start_utc="2026-03-09T13:00:00Z",
                     end_utc="2026-03-09T14:31:00Z")
    resp = _vote(client, "prop-1", "boss1", "approve")
    assert resp.status_code == 409
    violation, = resp.json()["violations"]
    assert violation["dependency_id"] == dep.json()["dependency_id"]
    proposal = _get_proposal(client, "prop-1")
    assert proposal["status"] == "pending"
    assert proposal["votes"] == []
    assert proposal["reservation"]["end_utc"] == "2026-03-09T14:00:00Z"
    events = _history(client, a["reservation_id"])
    assert [e["action"] for e in events] == ["created"]


def test_final_approve_within_dependency_succeeds(client):
    a = _reservation(client, "req-a", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", ("alice",))
    b = _reservation(client, "req-b", "2026-03-09T15:00:00Z",
                     "2026-03-09T16:00:00Z", ("bob",))
    client.post("/api/v1/reservation-dependencies", json={
        "dependency_request_id": "dep-1",
        "predecessor_id": a["reservation_id"],
        "successor_id": b["reservation_id"],
        "min_gap_minutes": 30,
    })
    # A 改到 13:00-14:30：14:30 + 30m = 15:00 <= B.start 15:00 → 满足
    _create_proposal(client, a["reservation_id"], approver_ids=["boss1"],
                     start_utc="2026-03-09T13:00:00Z",
                     end_utc="2026-03-09T14:30:00Z")
    resp = _vote(client, "prop-1", "boss1", "approve")
    assert resp.status_code == 201, resp.text
    assert resp.json()["status"] == "approved"
    assert resp.json()["reservation"]["end_utc"] == "2026-03-09T14:30:00Z"


def test_get_proposal_reflects_votes_and_reservation(client):
    r = _reservation(client)
    _create_proposal(client, r["reservation_id"])
    _vote(client, "prop-1", "boss1", "approve")
    data = _get_proposal(client, "prop-1")
    assert data["status"] == "pending"
    assert [v["approver_id"] for v in data["votes"]] == ["boss1"]
    assert data["reservation"]["version"] == 1
    _vote(client, "prop-1", "boss2", "approve")
    data = _get_proposal(client, "prop-1")
    assert data["status"] == "approved"
    assert len(data["votes"]) == 2
    assert data["reservation"]["version"] == 2
    assert data["reservation"]["start_utc"] == "2026-03-09T15:00:00Z"


# ------------------------------------------------------------------ 并发

def test_concurrent_final_votes_execute_reschedule_once(db_path):
    """同一必批人并发投最后一票：改期只提交一次，其余幂等返回。"""
    from app import reservations as reservation_core

    reservation, _ = reservation_core.create_reservation(
        request_id="req-1", start_utc=_utc(2026, 3, 9, 13),
        end_utc=_utc(2026, 3, 9, 14), participant_ids=["alice"],
        db_path=db_path)
    proposal, _, created = proposal_core.create_proposal(
        proposal_id="prop-1", reservation_id=reservation.id,
        expected_version=1, start_utc=_utc(2026, 3, 9, 15),
        end_utc=_utc(2026, 3, 9, 16), participant_ids=["alice"],
        approver_ids=["boss1"], db_path=db_path)
    assert created is True

    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        return proposal_core.vote_proposal(
            proposal_id="prop-1", approver_id="boss1",
            decision="approve", db_path=db_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda _: worker(), range(8)))

    assert sum(1 for _, _, recorded in outcomes if recorded) == 1  # 只投上一票
    assert {p.status for p, _, _ in outcomes} == {"approved"}
    # 重新读取预约：版本只推进一次
    conn = db_module.connect(db_path)
    try:
        row = conn.execute(
            "SELECT version, start_utc FROM reservations WHERE id = ?",
            (reservation.id,),
        ).fetchone()
        assert row["version"] == 2
        assert row["start_utc"] == "2026-03-09T15:00:00.000000Z"
        votes = conn.execute(
            "SELECT COUNT(*) AS n FROM reschedule_proposal_votes"
            " WHERE proposal_pk = ?",
            (proposal.id,),
        ).fetchone()
        assert votes["n"] == 1
        events = conn.execute(
            "SELECT action FROM reservation_events WHERE reservation_id = ?"
            " ORDER BY seq",
            (reservation.id,),
        ).fetchall()
        assert [e["action"] for e in events] == ["created", "rescheduled"]
    finally:
        conn.close()


def test_concurrent_distinct_approvers_single_reschedule(db_path):
    """两位必批人并发 approve：先到者落票 pending，后到者触发改期且仅一次。"""
    from app import reservations as reservation_core

    reservation, _ = reservation_core.create_reservation(
        request_id="req-1", start_utc=_utc(2026, 3, 9, 13),
        end_utc=_utc(2026, 3, 9, 14), participant_ids=["alice"],
        db_path=db_path)
    proposal_core.create_proposal(
        proposal_id="prop-1", reservation_id=reservation.id,
        expected_version=1, start_utc=_utc(2026, 3, 9, 15),
        end_utc=_utc(2026, 3, 9, 16), participant_ids=["alice"],
        approver_ids=["boss1", "boss2"], db_path=db_path)

    barrier = threading.Barrier(2)

    def worker(approver):
        barrier.wait()
        return proposal_core.vote_proposal(
            proposal_id="prop-1", approver_id=approver,
            decision="approve", db_path=db_path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(worker, ["boss1", "boss2"]))

    assert all(recorded for _, _, recorded in outcomes)
    # 写锁把两票串行化：先到者落票时必批人未齐（pending），
    # 后到者成为最后一票并触发改期（approved），改期只提交一次
    statuses = [p.status for p, _, _ in outcomes]
    assert statuses.count("approved") == 1
    assert statuses.count("pending") == 1
    conn = db_module.connect(db_path)
    try:
        row = conn.execute(
            "SELECT version FROM reservations WHERE id = ?",
            (reservation.id,),
        ).fetchone()
        assert row["version"] == 2  # 改期只提交一次
    finally:
        conn.close()


def test_concurrent_create_same_proposal_id_creates_single_proposal(db_path):
    """同一 proposal_id 并发创建：只新建一次，其余幂等返回。"""
    from app import reservations as reservation_core

    reservation, _ = reservation_core.create_reservation(
        request_id="req-1", start_utc=_utc(2026, 3, 9, 13),
        end_utc=_utc(2026, 3, 9, 14), participant_ids=["alice"],
        db_path=db_path)

    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        return proposal_core.create_proposal(
            proposal_id="prop-same", reservation_id=reservation.id,
            expected_version=1, start_utc=_utc(2026, 3, 9, 15),
            end_utc=_utc(2026, 3, 9, 16), participant_ids=["alice"],
            approver_ids=["boss1"], db_path=db_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda _: worker(), range(8)))

    ids = {p.id for p, _, _ in outcomes}
    assert len(ids) == 1
    assert sum(1 for _, _, created in outcomes if created) == 1


# ------------------------------------------------------------------ 迁移

def test_init_creates_proposal_tables_losslessly(tmp_path, monkeypatch):
    """旧版库（无提案表）初始化后新增提案/投票表，既有数据不动。"""
    import sqlite3

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

    # 新表立即可用：在旧库上创建提案并投票通过
    proposal, reservation, created = proposal_core.create_proposal(
        proposal_id="prop-1", reservation_id=1, expected_version=1,
        start_utc=_utc(2026, 3, 9, 15), end_utc=_utc(2026, 3, 9, 16),
        participant_ids=["alice"], approver_ids=["boss1"], db_path=path)
    assert created is True
    proposal, reservation, recorded = proposal_core.vote_proposal(
        proposal_id="prop-1", approver_id="boss1", decision="approve",
        db_path=path)
    assert recorded is True
    assert proposal.status == "approved"
    assert reservation.version == 2
