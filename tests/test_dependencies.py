"""预约时序依赖（reservation dependencies）测试。

覆盖：建边（201/幂等重放/同键异内容 409/自环 422/端点缺失 404/成环 409）、
改期与取消时在同一事务内重验全部可达依赖（违反 409 + 完整链路、
无部分写入、成功沿用版本与审计事件）、取消后前驱仍约束、
并发建边幂等、旧版库无损建新表。
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app import db as db_module
from app import dependencies as dependency_core
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


def _dep_payload(request_id, predecessor_id, successor_id, gap=30):
    return {
        "dependency_request_id": request_id,
        "predecessor_id": predecessor_id,
        "successor_id": successor_id,
        "min_gap_minutes": gap,
    }


def _create_pair(client, start_a="2026-03-09T13:00:00Z",
                 end_a="2026-03-09T14:00:00Z",
                 start_b="2026-03-09T15:00:00Z",
                 end_b="2026-03-09T16:00:00Z"):
    """创建两条不相交的预约 A、B（不同参与者，互不冲突）。"""
    a = _reservation(client, "req-a", start_a, end_a, ("alice",))
    b = _reservation(client, "req-b", start_b, end_b, ("bob",))
    return a, b


def _history(client, reservation_id):
    resp = client.get(f"/api/v1/reservations/{reservation_id}/history")
    assert resp.status_code == 200, resp.text
    return resp.json()["events"]


# ------------------------------------------------------------------ 建边

def test_create_dependency_returns_201(client):
    a, b = _create_pair(client)
    resp = client.post("/api/v1/reservation-dependencies",
                       json=_dep_payload("dep-1", a["reservation_id"],
                                         b["reservation_id"], 30))
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["dependency_id"] == 1
    assert data["dependency_request_id"] == "dep-1"
    assert data["predecessor_id"] == a["reservation_id"]
    assert data["successor_id"] == b["reservation_id"]
    assert data["min_gap_minutes"] == 30
    assert data["created_at"]


def test_create_dependency_idempotent_replay(client):
    a, b = _create_pair(client)
    payload = _dep_payload("dep-1", a["reservation_id"], b["reservation_id"], 45)
    first = client.post("/api/v1/reservation-dependencies", json=payload)
    assert first.status_code == 201
    replay = client.post("/api/v1/reservation-dependencies", json=payload)
    assert replay.status_code == 200
    assert replay.json() == first.json()  # 返回原依赖，不新增记录
    # 下一条依赖 id 紧接其后：确实只建了一条
    c = _reservation(client, "req-c", "2026-03-10T13:00:00Z",
                     "2026-03-10T14:00:00Z", ("carol",))
    nxt = client.post("/api/v1/reservation-dependencies",
                      json=_dep_payload("dep-2", b["reservation_id"],
                                        c["reservation_id"], 0))
    assert nxt.status_code == 201
    assert nxt.json()["dependency_id"] == first.json()["dependency_id"] + 1


@pytest.mark.parametrize("field", ["predecessor_id", "successor_id",
                                   "min_gap_minutes"])
def test_create_dependency_same_key_different_content_409(client, field):
    a, b = _create_pair(client)
    c = _reservation(client, "req-c", "2026-03-10T13:00:00Z",
                     "2026-03-10T14:00:00Z", ("carol",))
    payload = _dep_payload("dep-1", a["reservation_id"], b["reservation_id"], 30)
    assert client.post("/api/v1/reservation-dependencies",
                       json=payload).status_code == 201
    other = dict(payload)
    # 换一个端点（指向第三条预约，避免自环）或换间隔分钟：均为不同内容
    other[field] = (c["reservation_id"] if field != "min_gap_minutes" else 99)
    resp = client.post("/api/v1/reservation-dependencies", json=other)
    assert resp.status_code == 409
    data = resp.json()
    assert data["existing_dependency_id"] == 1
    assert "different content" in data["detail"]


def test_create_dependency_self_loop_rejected(client):
    a = _reservation(client, "req-a", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z")
    resp = client.post("/api/v1/reservation-dependencies",
                       json=_dep_payload("dep-1", a["reservation_id"],
                                         a["reservation_id"], 0))
    assert resp.status_code == 422


def test_create_dependency_core_self_loop_defense(db_path):
    """绕过 API 层直接调核心：自环按长度为 1 的环拒绝。"""
    from app import reservations as reservation_core

    reservation, _ = reservation_core.create_reservation(
        request_id="req-a", start_utc=_utc(2026, 3, 9, 13),
        end_utc=_utc(2026, 3, 9, 14), participant_ids=["alice"],
        db_path=db_path)
    with pytest.raises(dependency_core.DependencyCycleError):
        dependency_core.create_dependency(
            dependency_request_id="dep-1",
            predecessor_id=reservation.id, successor_id=reservation.id,
            min_gap_minutes=0, db_path=db_path)


def test_create_dependency_missing_reservation_404(client):
    a = _reservation(client, "req-a", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z")
    resp = client.post("/api/v1/reservation-dependencies",
                       json=_dep_payload("dep-1", a["reservation_id"], 999, 0))
    assert resp.status_code == 404
    assert resp.json()["detail"] == "reservation #999 not found"
    resp = client.post("/api/v1/reservation-dependencies",
                       json=_dep_payload("dep-2", 999, a["reservation_id"], 0))
    assert resp.status_code == 404


def test_create_dependency_negative_gap_rejected(client):
    a, b = _create_pair(client)
    resp = client.post("/api/v1/reservation-dependencies",
                       json=_dep_payload("dep-1", a["reservation_id"],
                                         b["reservation_id"], -1))
    assert resp.status_code == 422


def test_create_dependency_direct_cycle_rejected(client):
    a, b = _create_pair(client)
    assert client.post("/api/v1/reservation-dependencies",
                       json=_dep_payload("dep-1", a["reservation_id"],
                                         b["reservation_id"], 0)
                       ).status_code == 201
    resp = client.post("/api/v1/reservation-dependencies",
                       json=_dep_payload("dep-2", b["reservation_id"],
                                         a["reservation_id"], 0))
    assert resp.status_code == 409
    assert "cycle" in resp.json()["detail"]


def test_create_dependency_transitive_cycle_rejected(client):
    a, b = _create_pair(client)
    c = _reservation(client, "req-c", "2026-03-09T17:00:00Z",
                     "2026-03-09T18:00:00Z", ("carol",))
    aid, bid, cid = (r["reservation_id"] for r in (a, b, c))
    assert client.post("/api/v1/reservation-dependencies",
                       json=_dep_payload("dep-1", aid, bid, 0)
                       ).status_code == 201
    assert client.post("/api/v1/reservation-dependencies",
                       json=_dep_payload("dep-2", bid, cid, 0)
                       ).status_code == 201
    # C -> A 会闭合 A -> B -> C -> A 的环
    resp = client.post("/api/v1/reservation-dependencies",
                       json=_dep_payload("dep-3", cid, aid, 0))
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "cycle" in detail
    assert f"#{aid}" in detail and f"#{cid}" in detail
    # 成环被拒绝后未写入：重放 dep-3 仍按同样规则 409，而不是幂等返回
    assert client.post("/api/v1/reservation-dependencies",
                       json=_dep_payload("dep-3", cid, aid, 0)
                       ).status_code == 409


def test_create_dependency_allows_currently_violated_constraint(client):
    """建边只校验结构：约束当前不满足也允许建立（后续变更时统一强制）。"""
    a, b = _create_pair(client, start_b="2026-03-09T13:30:00Z",
                        end_b="2026-03-09T14:30:00Z")
    # A 13:00-14:00，B 13:30-14:30：A.end + 30m = 14:30 > B.start 13:30
    resp = client.post("/api/v1/reservation-dependencies",
                       json=_dep_payload("dep-1", a["reservation_id"],
                                         b["reservation_id"], 30))
    assert resp.status_code == 201, resp.text


# ------------------------------------------------------------------ 改期重验

def test_reschedule_within_constraint_succeeds(client):
    a, b = _create_pair(client)
    assert client.post("/api/v1/reservation-dependencies",
                       json=_dep_payload("dep-1", a["reservation_id"],
                                         b["reservation_id"], 30)
                       ).status_code == 201
    # A 改到 13:00-14:30：14:30 + 30m = 15:00 <= B.start 15:00（边界恰好满足）
    resp = client.put(f"/api/v1/reservations/{a['reservation_id']}", json={
        "expected_version": 1,
        "start_utc": "2026-03-09T13:00:00Z",
        "end_utc": "2026-03-09T14:30:00Z",
        "participant_ids": ["alice"],
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["version"] == 2
    assert data["end_utc"] == "2026-03-09T14:30:00Z"
    # 成功改期沿用现有审计事件：追加一条 rescheduled
    events = _history(client, a["reservation_id"])
    assert [e["action"] for e in events] == ["created", "rescheduled"]
    assert events[-1]["snapshot"]["version"] == 2


def test_reschedule_predecessor_violating_constraint_409(client):
    a, b = _create_pair(client)
    dep = client.post("/api/v1/reservation-dependencies",
                      json=_dep_payload("dep-1", a["reservation_id"],
                                        b["reservation_id"], 30)).json()
    # A 改到 13:00-14:31：14:31 + 30m = 15:01 > B.start 15:00 → 违反
    resp = client.put(f"/api/v1/reservations/{a['reservation_id']}", json={
        "expected_version": 1,
        "start_utc": "2026-03-09T13:00:00Z",
        "end_utc": "2026-03-09T14:31:00Z",
        "participant_ids": ["alice"],
    })
    assert resp.status_code == 409
    data = resp.json()
    assert "dependency violation" in data["detail"]
    violation, = data["violations"]
    assert violation["dependency_id"] == dep["dependency_id"]
    assert violation["predecessor_id"] == a["reservation_id"]
    assert violation["successor_id"] == b["reservation_id"]
    assert violation["min_gap_minutes"] == 30
    assert violation["predecessor_end_utc"] == "2026-03-09T14:31:00Z"
    assert violation["successor_start_utc"] == "2026-03-09T15:00:00Z"
    assert violation["required_successor_start_utc"] == "2026-03-09T15:01:00Z"
    # 完整链路：被变更预约直连的这条边
    assert violation["chain"] == [{
        "dependency_id": dep["dependency_id"],
        "predecessor_id": a["reservation_id"],
        "successor_id": b["reservation_id"],
        "min_gap_minutes": 30,
    }]
    # 不能写入部分变更：版本、时段、参与者、审计事件均保持原状
    events = _history(client, a["reservation_id"])
    assert [e["action"] for e in events] == ["created"]
    snapshot = events[0]["snapshot"]
    assert snapshot["version"] == 1
    assert snapshot["end_utc"] == "2026-03-09T14:00:00Z"
    # 版本未推进：按原版本号重试合法改期仍可成功
    ok = client.put(f"/api/v1/reservations/{a['reservation_id']}", json={
        "expected_version": 1,
        "start_utc": "2026-03-09T13:00:00Z",
        "end_utc": "2026-03-09T14:30:00Z",
        "participant_ids": ["alice"],
    })
    assert ok.status_code == 200, ok.text


def test_reschedule_successor_earlier_violating_constraint_409(client):
    a, b = _create_pair(client)
    assert client.post("/api/v1/reservation-dependencies",
                       json=_dep_payload("dep-1", a["reservation_id"],
                                         b["reservation_id"], 30)
                       ).status_code == 201
    # B 提前到 14:00-15:00：A.end 14:00 + 30m = 14:30 > 14:00 → 违反
    resp = client.put(f"/api/v1/reservations/{b['reservation_id']}", json={
        "expected_version": 1,
        "start_utc": "2026-03-09T14:00:00Z",
        "end_utc": "2026-03-09T15:00:00Z",
        "participant_ids": ["bob"],
    })
    assert resp.status_code == 409
    violation, = resp.json()["violations"]
    assert violation["successor_id"] == b["reservation_id"]
    assert violation["required_successor_start_utc"] == "2026-03-09T14:30:00Z"
    # B 保持原状
    events = _history(client, b["reservation_id"])
    assert [e["action"] for e in events] == ["created"]


def test_reschedule_revalidates_transitive_dependencies(client):
    """可达依赖整图重验：被改预约未直接参与的深层边违反也会 409。"""
    a, b = _create_pair(client)
    c = _reservation(client, "req-c", "2026-03-09T15:30:00Z",
                     "2026-03-09T16:30:00Z", ("carol",))
    aid, bid, cid = (r["reservation_id"] for r in (a, b, c))
    dep_ab = client.post("/api/v1/reservation-dependencies",
                         json=_dep_payload("dep-1", aid, bid, 30)).json()
    # B(15:00-16:00) -> C(15:30-16:30) gap 0：建边时即不满足（允许建立）
    dep_bc = client.post("/api/v1/reservation-dependencies",
                         json=_dep_payload("dep-2", bid, cid, 0)).json()
    # A 的改期本身满足 A->B（14:30 + 30m = 15:00 <= 15:00），
    # 但可达子图中 B->C 已违反 → 409，链路为 A->B、B->C 完整路径
    resp = client.put(f"/api/v1/reservations/{aid}", json={
        "expected_version": 1,
        "start_utc": "2026-03-09T13:00:00Z",
        "end_utc": "2026-03-09T14:30:00Z",
        "participant_ids": ["alice"],
    })
    assert resp.status_code == 409
    violation, = resp.json()["violations"]
    assert violation["dependency_id"] == dep_bc["dependency_id"]
    assert violation["predecessor_id"] == bid
    assert violation["successor_id"] == cid
    assert violation["chain"] == [
        {"dependency_id": dep_ab["dependency_id"], "predecessor_id": aid,
         "successor_id": bid, "min_gap_minutes": 30},
        {"dependency_id": dep_bc["dependency_id"], "predecessor_id": bid,
         "successor_id": cid, "min_gap_minutes": 0},
    ]
    # A 未被改动
    assert [e["action"] for e in _history(client, aid)] == ["created"]


def test_reschedule_lists_all_violations(client):
    """一次改期同时违反多条可达依赖：全部列出。"""
    a, b = _create_pair(client)
    c = _reservation(client, "req-c", "2026-03-09T17:00:00Z",
                     "2026-03-09T18:00:00Z", ("carol",))
    aid, bid, cid = (r["reservation_id"] for r in (a, b, c))
    client.post("/api/v1/reservation-dependencies",
                json=_dep_payload("dep-1", aid, bid, 30))
    client.post("/api/v1/reservation-dependencies",
                json=_dep_payload("dep-2", bid, cid, 30))
    # B 改到 13:30-16:45：A.end 14:00 + 30m = 14:30 > 13:30（违反 A->B），
    # B.end 16:45 + 30m = 17:15 > C.start 17:00（违反 B->C）
    resp = client.put(f"/api/v1/reservations/{bid}", json={
        "expected_version": 1,
        "start_utc": "2026-03-09T13:30:00Z",
        "end_utc": "2026-03-09T16:45:00Z",
        "participant_ids": ["bob"],
    })
    assert resp.status_code == 409
    violations = resp.json()["violations"]
    assert {v["dependency_id"] for v in violations} == {1, 2}
    assert [e["action"] for e in _history(client, bid)] == ["created"]


def test_reschedule_unrelated_reservation_not_affected(client):
    """存在被违反的依赖不影响无依赖关系的其他预约。"""
    a, b = _create_pair(client, start_b="2026-03-09T13:30:00Z",
                        end_b="2026-03-09T14:30:00Z")
    # A -> B 当前即被违反（允许建立）
    client.post("/api/v1/reservation-dependencies",
                json=_dep_payload("dep-1", a["reservation_id"],
                                  b["reservation_id"], 30))
    c = _reservation(client, "req-c", "2026-03-10T13:00:00Z",
                     "2026-03-10T14:00:00Z", ("carol",))
    resp = client.put(f"/api/v1/reservations/{c['reservation_id']}", json={
        "expected_version": 1,
        "start_utc": "2026-03-10T15:00:00Z",
        "end_utc": "2026-03-10T16:00:00Z",
        "participant_ids": ["carol"],
    })
    assert resp.status_code == 200, resp.text


def test_reschedule_can_repair_violated_constraint(client):
    """改期把已违反的约束修复为满足：重验通过，正常提交。"""
    a, b = _create_pair(client, start_b="2026-03-09T13:30:00Z",
                        end_b="2026-03-09T14:30:00Z")
    client.post("/api/v1/reservation-dependencies",
                json=_dep_payload("dep-1", a["reservation_id"],
                                  b["reservation_id"], 30))
    # B 改到 15:00-16:00：A.end 14:00 + 30m = 15:00 <= 15:00 → 修复
    resp = client.put(f"/api/v1/reservations/{b['reservation_id']}", json={
        "expected_version": 1,
        "start_utc": "2026-03-09T15:00:00Z",
        "end_utc": "2026-03-09T16:00:00Z",
        "participant_ids": ["bob"],
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["version"] == 2


def test_zero_gap_allows_adjacent_but_not_overlap(client):
    a, b = _create_pair(client)
    assert client.post("/api/v1/reservation-dependencies",
                       json=_dep_payload("dep-1", a["reservation_id"],
                                         b["reservation_id"], 0)
                       ).status_code == 201
    # B 改到紧邻 A 之后（14:00-15:00）：A.end 14:00 + 0m <= 14:00 → 满足
    resp = client.put(f"/api/v1/reservations/{b['reservation_id']}", json={
        "expected_version": 1,
        "start_utc": "2026-03-09T14:00:00Z",
        "end_utc": "2026-03-09T15:00:00Z",
        "participant_ids": ["bob"],
    })
    assert resp.status_code == 200, resp.text
    # 再提前 1 分钟（13:59 开始）→ 违反
    resp = client.put(f"/api/v1/reservations/{b['reservation_id']}", json={
        "expected_version": 2,
        "start_utc": "2026-03-09T13:59:00Z",
        "end_utc": "2026-03-09T15:00:00Z",
        "participant_ids": ["bob"],
    })
    assert resp.status_code == 409
    violation, = resp.json()["violations"]
    assert violation["required_successor_start_utc"] == "2026-03-09T14:00:00Z"


# ------------------------------------------------------------------ 取消重验

def test_cancel_with_satisfied_dependencies_succeeds(client):
    a, b = _create_pair(client)
    client.post("/api/v1/reservation-dependencies",
                json=_dep_payload("dep-1", a["reservation_id"],
                                  b["reservation_id"], 30))
    resp = client.delete(
        f"/api/v1/reservations/{a['reservation_id']}?expected_version=1")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "canceled"
    assert data["version"] == 2
    events = _history(client, a["reservation_id"])
    assert [e["action"] for e in events] == ["created", "canceled"]


def test_cancel_with_violated_dependency_409(client):
    """取消也在同一事务重验可达依赖：当前被违反 → 409，不写部分变更。"""
    a, b = _create_pair(client, start_b="2026-03-09T13:30:00Z",
                        end_b="2026-03-09T14:30:00Z")
    # A(13:00-14:00) -> B(13:30-14:30) gap 30：建边时即被违反
    dep = client.post("/api/v1/reservation-dependencies",
                      json=_dep_payload("dep-1", a["reservation_id"],
                                        b["reservation_id"], 30)).json()
    resp = client.delete(
        f"/api/v1/reservations/{a['reservation_id']}?expected_version=1")
    assert resp.status_code == 409
    violation, = resp.json()["violations"]
    assert violation["dependency_id"] == dep["dependency_id"]
    # 后继端取消同样被重验拦截
    resp = client.delete(
        f"/api/v1/reservations/{b['reservation_id']}?expected_version=1")
    assert resp.status_code == 409
    # 两条预约均保持 active、版本未推进、无 canceled 事件
    for r in (a, b):
        events = _history(client, r["reservation_id"])
        assert [e["action"] for e in events] == ["created"]
        assert events[0]["snapshot"]["status"] == "active"
    # 修复约束（B 改到 15:00-16:00）后即可正常取消
    ok = client.put(f"/api/v1/reservations/{b['reservation_id']}", json={
        "expected_version": 1,
        "start_utc": "2026-03-09T15:00:00Z",
        "end_utc": "2026-03-09T16:00:00Z",
        "participant_ids": ["bob"],
    })
    assert ok.status_code == 200, ok.text
    resp = client.delete(
        f"/api/v1/reservations/{a['reservation_id']}?expected_version=1")
    assert resp.status_code == 200, resp.text


def test_canceled_predecessor_still_constrains_successor(client):
    """取消是软删除：已取消前驱的依赖仍按其存储时段约束后继。"""
    a, b = _create_pair(client)
    client.post("/api/v1/reservation-dependencies",
                json=_dep_payload("dep-1", a["reservation_id"],
                                  b["reservation_id"], 30))
    resp = client.delete(
        f"/api/v1/reservations/{a['reservation_id']}?expected_version=1")
    assert resp.status_code == 200, resp.text
    # B 提前到 14:00-15:00：仍受已取消 A 的 14:00 + 30m 约束 → 409
    resp = client.put(f"/api/v1/reservations/{b['reservation_id']}", json={
        "expected_version": 1,
        "start_utc": "2026-03-09T14:00:00Z",
        "end_utc": "2026-03-09T15:00:00Z",
        "participant_ids": ["bob"],
    })
    assert resp.status_code == 409
    violation, = resp.json()["violations"]
    assert violation["predecessor_id"] == a["reservation_id"]


def test_dependency_may_reference_canceled_reservation(client):
    """已取消预约仍存在（软删除），可以作为依赖端点。"""
    a, b = _create_pair(client)
    assert client.delete(
        f"/api/v1/reservations/{a['reservation_id']}?expected_version=1"
    ).status_code == 200
    resp = client.post("/api/v1/reservation-dependencies",
                       json=_dep_payload("dep-1", a["reservation_id"],
                                         b["reservation_id"], 30))
    assert resp.status_code == 201, resp.text


# ------------------------------------------------------------------ 并发与迁移

def test_concurrent_create_same_key_creates_single_dependency(db_path):
    """同一 dependency_request_id 并发建边：只新建一次，其余幂等返回。"""
    barrier = threading.Barrier(8)
    kwargs = dict(
        request_id="req-a", start_utc=_utc(2026, 3, 9, 13),
        end_utc=_utc(2026, 3, 9, 14), participant_ids=["alice"],
        db_path=db_path)
    from app import reservations as reservation_core
    a, _ = reservation_core.create_reservation(**kwargs)
    b, _ = reservation_core.create_reservation(
        request_id="req-b", start_utc=_utc(2026, 3, 9, 15),
        end_utc=_utc(2026, 3, 9, 16), participant_ids=["bob"],
        db_path=db_path)

    def worker():
        barrier.wait()
        return dependency_core.create_dependency(
            dependency_request_id="dep-same",
            predecessor_id=a.id, successor_id=b.id,
            min_gap_minutes=30, db_path=db_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda _: worker(), range(8)))

    ids = {d.id for d, _ in outcomes}
    assert len(ids) == 1                                      # 同一条依赖
    assert sum(1 for _, created in outcomes if created) == 1  # 只新建一次
    # 数据库里确实只有这一条：下一条依赖 id 紧接其后
    c, _ = reservation_core.create_reservation(
        request_id="req-c", start_utc=_utc(2026, 3, 10, 13),
        end_utc=_utc(2026, 3, 10, 14), participant_ids=["carol"],
        db_path=db_path)
    nxt, created = dependency_core.create_dependency(
        dependency_request_id="dep-next",
        predecessor_id=b.id, successor_id=c.id,
        min_gap_minutes=0, db_path=db_path)
    assert created is True
    assert nxt.id == ids.pop() + 1


def test_init_creates_dependency_table_losslessly(tmp_path, monkeypatch):
    """旧版库（无依赖表）初始化后新增依赖表，既有数据不动。"""
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
        assert "reservation_dependencies" in names
        # 既有预约无损
        assert conn.execute(
            "SELECT request_id, participants_key FROM reservations"
        ).fetchone() == ("old", "[\"alice\"]")
    finally:
        conn.close()

    # 新表立即可用：在旧库上建立依赖
    from app import reservations as reservation_core
    b, _ = reservation_core.create_reservation(
        request_id="req-b", start_utc=_utc(2026, 3, 9, 15),
        end_utc=_utc(2026, 3, 9, 16), participant_ids=["bob"],
        db_path=path)
    dep, created = dependency_core.create_dependency(
        dependency_request_id="dep-1", predecessor_id=1,
        successor_id=b.id, min_gap_minutes=30, db_path=path)
    assert created is True
    assert dep.predecessor_id == 1
