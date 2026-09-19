"""会议室批量分配测试。

覆盖：会议室 upsert（201 创建 / 200 更新 / 设备去重排序 / 校验 422）、
查询（单个 404、列表过滤）、批量分配（确定性最优解：总空余座位最少 +
（预约 id, room_id）字典序、已有分配占用、半开区间相邻不冲突、
容量/设备过滤、批内互斥、无解 409 列出无法分配预约且已有分配不变、
batch_id 幂等重放 / 异内容 409、预约不存在 404 / 已取消 409、
重批替换旧分配）、分配查询（列表 / 过滤 / 按批次 404）、
改期/取消/交换释放分配、并发批次恰一个赢家、旧版库无损建新表。
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app import db as db_module
from app import room_assignments as room_core
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


def _room(client, room_id, capacity, equipment=()):
    resp = client.put(f"/api/v1/meeting-rooms/{room_id}", json={
        "capacity": capacity,
        "equipment": list(equipment),
    })
    assert resp.status_code in (200, 201), resp.text
    return resp.json()


def _reservation(client, request_id, start, end, participants=("alice",)):
    resp = client.post("/api/v1/reservations", json={
        "request_id": request_id,
        "start_utc": start,
        "end_utc": end,
        "participant_ids": list(participants),
    })
    assert resp.status_code == 201, resp.text
    return resp.json()


def _item(reservation_id, capacity, equipment=()):
    return {
        "reservation_id": reservation_id,
        "required_capacity": capacity,
        "required_equipment": list(equipment),
    }


def _batch(client, batch_id, items, expect=201):
    resp = client.post("/api/v1/room-assignments",
                       json={"batch_id": batch_id, "items": items})
    assert resp.status_code == expect, resp.text
    return resp.json()


def _assignments(client, **params):
    resp = client.get("/api/v1/room-assignments", params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ------------------------------------------------------------------ 会议室

def test_upsert_room_creates_201(client):
    resp = client.put("/api/v1/meeting-rooms/room-a", json={
        "capacity": 6,
        "equipment": ["whiteboard", "tv", "tv", "whiteboard"],
    })
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["room_id"] == "room-a"
    assert data["capacity"] == 6
    # 设备标签去重并按字典序排序
    assert data["equipment"] == ["tv", "whiteboard"]
    assert data["created_at"] == data["updated_at"]


def test_upsert_room_updates_200(client):
    first = _room(client, "room-a", 6, ["tv"])
    resp = client.put("/api/v1/meeting-rooms/room-a", json={
        "capacity": 10,
        "equipment": ["projector"],
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["capacity"] == 10
    assert data["equipment"] == ["projector"]
    assert data["created_at"] == first["created_at"]  # 创建时刻保留
    # GET 反映最新状态
    got = client.get("/api/v1/meeting-rooms/room-a")
    assert got.status_code == 200
    assert got.json() == data


@pytest.mark.parametrize("payload", [
    {"capacity": 0, "equipment": []},
    {"capacity": -3, "equipment": []},
    {"capacity": 4, "equipment": [""]},
    {"capacity": 4, "equipment": ["   "]},
])
def test_upsert_room_validation_422(client, payload):
    resp = client.put("/api/v1/meeting-rooms/room-a", json=payload)
    assert resp.status_code == 422


def test_get_room_404(client):
    resp = client.get("/api/v1/meeting-rooms/nope")
    assert resp.status_code == 404


def test_list_rooms_sorted_and_filtered(client):
    _room(client, "room-b", 8, ["tv"])
    _room(client, "room-a", 4, ["tv", "whiteboard"])
    _room(client, "room-c", 2, [])

    resp = client.get("/api/v1/meeting-rooms")
    assert resp.status_code == 200
    assert [r["room_id"] for r in resp.json()] == ["room-a", "room-b", "room-c"]

    by_cap = client.get("/api/v1/meeting-rooms", params={"min_capacity": 4})
    assert [r["room_id"] for r in by_cap.json()] == ["room-a", "room-b"]

    by_eq = client.get("/api/v1/meeting-rooms", params={"equipment": "tv"})
    assert [r["room_id"] for r in by_eq.json()] == ["room-a", "room-b"]

    both = client.get("/api/v1/meeting-rooms",
                      params={"equipment": "tv,whiteboard"})
    assert [r["room_id"] for r in both.json()] == ["room-a"]

    combo = client.get("/api/v1/meeting-rooms",
                       params={"min_capacity": 4, "equipment": "whiteboard"})
    assert [r["room_id"] for r in combo.json()] == ["room-a"]


# ---------------------------------------------------------------- 批量分配

def test_batch_prefers_min_total_spare(client):
    """总空余座位最少优先：即使更紧的房间 room_id 更大也选它。"""
    _room(client, "a-big", 10)
    _room(client, "z-small", 4)
    res = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                       "2026-03-09T14:00:00Z")
    data = _batch(client, "batch-1", [_item(res["reservation_id"], 3)])
    assert data["batch_id"] == "batch-1"
    assert data["total_spare_seats"] == 1
    assert len(data["assignments"]) == 1
    a = data["assignments"][0]
    assert a["room_id"] == "z-small"  # 空余 1 < 7，字典序让位于空余
    assert a["reservation_id"] == res["reservation_id"]
    assert a["required_capacity"] == 3
    assert a["room_capacity"] == 4
    assert a["spare_seats"] == 1
    assert a["start_utc"] == "2026-03-09T13:00:00Z"
    assert a["end_utc"] == "2026-03-09T14:00:00Z"


def test_batch_lex_tiebreak_single_reservation(client):
    """空余座位并列时按 room_id 字典序取解。"""
    _room(client, "room-b", 4)
    _room(client, "room-a", 4)
    res = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                       "2026-03-09T14:00:00Z")
    data = _batch(client, "batch-1", [_item(res["reservation_id"], 3)])
    assert data["assignments"][0]["room_id"] == "room-a"
    assert data["total_spare_seats"] == 1


def test_batch_lex_tiebreak_vector(client):
    """多条预约并列时按（预约 id 升序, room_id）向量字典序取解。"""
    _room(client, "room-a", 4)
    _room(client, "room-b", 4)
    res1 = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                        "2026-03-09T14:00:00Z", ("alice",))
    res2 = _reservation(client, "req-2", "2026-03-09T13:00:00Z",
                        "2026-03-09T14:00:00Z", ("bob",))
    data = _batch(client, "batch-1", [
        _item(res2["reservation_id"], 3),
        _item(res1["reservation_id"], 3),
    ])
    # (room-a, room-b) 与 (room-b, room-a) 空余相同，取向量字典序最小者
    assert [(a["reservation_id"], a["room_id"]) for a in data["assignments"]] == [
        (res1["reservation_id"], "room-a"),
        (res2["reservation_id"], "room-b"),
    ]
    assert data["total_spare_seats"] == 2


def test_batch_respects_existing_assignments(client):
    """已有分配占用房间；半开区间相邻（end == start）不冲突。"""
    _room(client, "room-1", 4)
    _room(client, "room-2", 4)
    res1 = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                        "2026-03-09T14:00:00Z", ("alice",))
    res2 = _reservation(client, "req-2", "2026-03-09T13:30:00Z",
                        "2026-03-09T14:30:00Z", ("bob",))
    res3 = _reservation(client, "req-3", "2026-03-09T14:00:00Z",
                        "2026-03-09T15:00:00Z", ("carol",))
    _batch(client, "batch-1", [_item(res1["reservation_id"], 3)])
    # res2 与 res1 时段重叠：room-1 被占，落到 room-2
    data2 = _batch(client, "batch-2", [_item(res2["reservation_id"], 3)])
    assert data2["assignments"][0]["room_id"] == "room-2"
    # res3 与 res1 首尾相接（不冲突），但与 res2 重叠 → room-1 可用
    data3 = _batch(client, "batch-3", [_item(res3["reservation_id"], 3)])
    assert data3["assignments"][0]["room_id"] == "room-1"


def test_batch_equipment_and_capacity_filters(client):
    """设备须全覆盖、容量须不小于所需；不满足的房间不可选。"""
    _room(client, "room-tv", 4, ["tv"])
    _room(client, "room-plain", 8, [])
    res1 = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                        "2026-03-09T14:00:00Z", ("alice",))
    res2 = _reservation(client, "req-2", "2026-03-09T13:00:00Z",
                        "2026-03-09T14:00:00Z", ("bob",))
    data = _batch(client, "batch-1", [
        _item(res1["reservation_id"], 3, ["tv"]),
        _item(res2["reservation_id"], 6),
    ])
    by_id = {a["reservation_id"]: a["room_id"] for a in data["assignments"]}
    assert by_id[res1["reservation_id"]] == "room-tv"      # 需要 tv
    assert by_id[res2["reservation_id"]] == "room-plain"   # room-tv 容量不足


def test_batch_unsolvable_lists_unassignable_and_keeps_state(client):
    """整批无解 → 409 + 无法分配预约；已有分配与批次均不变。"""
    _room(client, "room-1", 4)
    res0 = _reservation(client, "req-0", "2026-03-09T09:00:00Z",
                        "2026-03-09T10:00:00Z", ("carol",))
    res1 = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                        "2026-03-09T14:00:00Z", ("alice",))
    res2 = _reservation(client, "req-2", "2026-03-09T13:00:00Z",
                        "2026-03-09T14:00:00Z", ("bob",))
    _batch(client, "batch-0", [_item(res0["reservation_id"], 3)])

    resp = client.post("/api/v1/room-assignments", json={
        "batch_id": "batch-bad",
        "items": [_item(res1["reservation_id"], 3),
                  _item(res2["reservation_id"], 3)],
    })
    assert resp.status_code == 409, resp.text
    body = resp.json()
    # 确定性最优部分解：预约 id 较小者获得房间，较大者无法分配
    assert body["unassignable"] == [res2["reservation_id"]]
    # 已有分配不变、失败批次未落库
    assert [(a["reservation_id"], a["room_id"]) for a in _assignments(client)] == [
        (res0["reservation_id"], "room-1"),
    ]
    assert client.get("/api/v1/room-assignments/batch-bad").status_code == 404


def test_batch_unassignable_when_no_room_fits(client):
    """容量不足 / 设备缺失导致单条预约无处可去 → 409 列出该预约。"""
    _room(client, "room-1", 2, ["tv"])
    res1 = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                        "2026-03-09T14:00:00Z", ("alice",))
    res2 = _reservation(client, "req-2", "2026-03-09T15:00:00Z",
                        "2026-03-09T16:00:00Z", ("bob",))
    resp = client.post("/api/v1/room-assignments", json={
        "batch_id": "batch-bad",
        "items": [_item(res1["reservation_id"], 8),
                  _item(res2["reservation_id"], 1, ["telepresence"])],
    })
    assert resp.status_code == 409
    assert resp.json()["unassignable"] == [
        res1["reservation_id"], res2["reservation_id"]]


def test_batch_unassignable_deterministic_max_subset(client):
    """多解时报告确定性最优部分解的未分配项（未分配排在任何房间之后）。"""
    _room(client, "room-a", 2)
    _room(client, "room-b", 2)
    made = [
        _reservation(client, f"req-{i}", "2026-03-09T13:00:00Z",
                     "2026-03-09T14:00:00Z", (f"p{i}",))
        for i in range(1, 4)
    ]
    resp = client.post("/api/v1/room-assignments", json={
        "batch_id": "batch-bad",
        "items": [_item(r["reservation_id"], 2) for r in made],
    })
    assert resp.status_code == 409
    # 三条预约两间房：id 最小的两条获得房间，最大的一条无法分配
    assert resp.json()["unassignable"] == [made[2]["reservation_id"]]


def test_batch_idempotent_replay(client):
    _room(client, "room-1", 4)
    res = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                       "2026-03-09T14:00:00Z")
    payload = {"batch_id": "batch-1",
               "items": [_item(res["reservation_id"], 3)]}
    first = client.post("/api/v1/room-assignments", json=payload)
    assert first.status_code == 201
    replay = client.post("/api/v1/room-assignments", json=payload)
    assert replay.status_code == 200
    assert replay.json() == first.json()  # 返回同一结果，不新增记录
    assert len(_assignments(client)) == 1


def test_batch_same_id_different_content_409(client):
    _room(client, "room-1", 4)
    res = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                       "2026-03-09T14:00:00Z")
    _batch(client, "batch-1", [_item(res["reservation_id"], 3)])
    resp = client.post("/api/v1/room-assignments", json={
        "batch_id": "batch-1",
        "items": [_item(res["reservation_id"], 2)],  # 同键不同内容
    })
    assert resp.status_code == 409
    assert resp.json()["existing_batch_id"] == "batch-1"


def test_batch_reservation_not_found_404(client):
    _room(client, "room-1", 4)
    resp = client.post("/api/v1/room-assignments", json={
        "batch_id": "batch-1",
        "items": [_item(999, 3)],
    })
    assert resp.status_code == 404


def test_batch_canceled_reservation_409(client):
    _room(client, "room-1", 4)
    res = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                       "2026-03-09T14:00:00Z")
    resp = client.delete(
        f"/api/v1/reservations/{res['reservation_id']}?expected_version=1")
    assert resp.status_code == 200
    resp = client.post("/api/v1/room-assignments", json={
        "batch_id": "batch-1",
        "items": [_item(res["reservation_id"], 3)],
    })
    assert resp.status_code == 409
    assert "current_version" in resp.json()


@pytest.mark.parametrize("items", [
    [],  # 空批次
    [{"reservation_id": 1, "required_capacity": 0}],
    [{"reservation_id": 1, "required_capacity": 2},
     {"reservation_id": 1, "required_capacity": 3}],  # 预约重复
    [{"reservation_id": 1, "required_capacity": 2,
      "required_equipment": [" "]}],
])
def test_batch_validation_422(client, items):
    resp = client.post("/api/v1/room-assignments",
                       json={"batch_id": "batch-1", "items": items})
    assert resp.status_code == 422


def test_batch_replaces_member_assignment(client):
    """重新提交包含某预约的新批次：同事务替换其旧分配。"""
    _room(client, "room-1", 4)
    _room(client, "room-2", 4)
    res1 = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                        "2026-03-09T14:00:00Z", ("alice",))
    res2 = _reservation(client, "req-2", "2026-03-09T13:00:00Z",
                        "2026-03-09T14:00:00Z", ("bob",))
    _batch(client, "batch-1", [_item(res1["reservation_id"], 3)])
    # 第二批包含 res1（旧分配被替换，不再占用 room-1）与 res2
    data = _batch(client, "batch-2", [
        _item(res1["reservation_id"], 3),
        _item(res2["reservation_id"], 3),
    ])
    assert [(a["reservation_id"], a["room_id"]) for a in data["assignments"]] == [
        (res1["reservation_id"], "room-1"),
        (res2["reservation_id"], "room-2"),
    ]
    current = _assignments(client)
    assert len(current) == 2
    assert all(a["batch_id"] == "batch-2" for a in current)
    # 第一批的分配已被替换：按批次查询反映最新状态
    batch1 = client.get("/api/v1/room-assignments/batch-1")
    assert batch1.status_code == 200
    assert batch1.json()["assignments"] == []


def test_failed_batch_keeps_existing_assignment(client):
    """重批失败时旧分配原样保留（不解已有分配）。"""
    _room(client, "room-1", 4)
    res1 = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                        "2026-03-09T14:00:00Z", ("alice",))
    res2 = _reservation(client, "req-2", "2026-03-09T13:00:00Z",
                        "2026-03-09T14:00:00Z", ("bob",))
    _batch(client, "batch-1", [_item(res1["reservation_id"], 3)])
    resp = client.post("/api/v1/room-assignments", json={
        "batch_id": "batch-2",
        "items": [_item(res1["reservation_id"], 3),
                  _item(res2["reservation_id"], 3)],
    })
    assert resp.status_code == 409
    assert resp.json()["unassignable"] == [res2["reservation_id"]]
    current = _assignments(client)
    assert [(a["batch_id"], a["reservation_id"], a["room_id"])
            for a in current] == [("batch-1", res1["reservation_id"], "room-1")]


# ---------------------------------------------------------------- 分配查询

def test_list_assignments_sorted_and_filtered(client):
    _room(client, "room-1", 4)
    _room(client, "room-2", 4)
    res1 = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                        "2026-03-09T14:00:00Z", ("alice",))
    res2 = _reservation(client, "req-2", "2026-03-09T15:00:00Z",
                        "2026-03-09T16:00:00Z", ("bob",))
    _batch(client, "batch-1", [_item(res2["reservation_id"], 3)])
    _batch(client, "batch-2", [_item(res1["reservation_id"], 3)])

    all_rows = _assignments(client)
    # 按预约 id 升序；两条预约不同时段，字典序最小解都落在 room-1
    assert [(a["reservation_id"], a["room_id"], a["batch_id"])
            for a in all_rows] == [
        (res1["reservation_id"], "room-1", "batch-2"),
        (res2["reservation_id"], "room-1", "batch-1"),
    ]
    assert all_rows[0]["spare_seats"] == 1

    by_room = _assignments(client, room_id="room-1")
    assert len(by_room) == 2
    assert _assignments(client, room_id="room-2") == []

    by_res = _assignments(client, reservation_id=res2["reservation_id"])
    assert [a["batch_id"] for a in by_res] == ["batch-1"]


def test_get_batch_404_and_200(client):
    assert client.get("/api/v1/room-assignments/nope").status_code == 404
    _room(client, "room-1", 4)
    res = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                       "2026-03-09T14:00:00Z")
    created = _batch(client, "batch-1", [_item(res["reservation_id"], 3)])
    got = client.get("/api/v1/room-assignments/batch-1")
    assert got.status_code == 200
    assert got.json() == created


# ------------------------------------------------------- 预约生命周期联动

def test_cancel_releases_assignment(client):
    """取消预约同事务释放会议室分配，房间立即可被新批次使用。"""
    _room(client, "room-1", 4)
    res1 = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                        "2026-03-09T14:00:00Z", ("alice",))
    _batch(client, "batch-1", [_item(res1["reservation_id"], 3)])
    resp = client.delete(
        f"/api/v1/reservations/{res1['reservation_id']}?expected_version=1")
    assert resp.status_code == 200
    assert _assignments(client) == []
    # 同时段新预约可以拿到同一房间
    res2 = _reservation(client, "req-2", "2026-03-09T13:00:00Z",
                        "2026-03-09T14:00:00Z", ("bob",))
    data = _batch(client, "batch-2", [_item(res2["reservation_id"], 3)])
    assert data["assignments"][0]["room_id"] == "room-1"


def test_reschedule_releases_assignment(client):
    """改期同事务释放会议室分配（分配与求解时的时段绑定）。"""
    _room(client, "room-1", 4)
    res1 = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                        "2026-03-09T14:00:00Z", ("alice",))
    _batch(client, "batch-1", [_item(res1["reservation_id"], 3)])
    resp = client.put(f"/api/v1/reservations/{res1['reservation_id']}", json={
        "expected_version": 1,
        "start_utc": "2026-03-09T15:00:00Z",
        "end_utc": "2026-03-09T16:00:00Z",
        "participant_ids": ["alice"],
    })
    assert resp.status_code == 200, resp.text
    assert _assignments(client) == []
    # 原时段空出：新预约可以分配到 room-1
    res2 = _reservation(client, "req-2", "2026-03-09T13:00:00Z",
                        "2026-03-09T14:00:00Z", ("bob",))
    data = _batch(client, "batch-2", [_item(res2["reservation_id"], 3)])
    assert data["assignments"][0]["room_id"] == "room-1"


def test_swap_releases_assignments(client):
    """时段交换同事务释放双方的会议室分配。"""
    _room(client, "room-1", 4)
    res1 = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                        "2026-03-09T14:00:00Z", ("alice",))
    res2 = _reservation(client, "req-2", "2026-03-09T16:00:00Z",
                        "2026-03-09T17:00:00Z", ("bob",))
    data = _batch(client, "batch-1", [
        _item(res1["reservation_id"], 3),
        _item(res2["reservation_id"], 3),
    ])
    assert len(data["assignments"]) == 2
    resp = client.post("/api/v1/reservations/swap", json={
        "first": {"reservation_id": res1["reservation_id"],
                  "expected_version": 1},
        "second": {"reservation_id": res2["reservation_id"],
                   "expected_version": 1},
    })
    assert resp.status_code == 200, resp.text
    assert _assignments(client) == []


def test_failed_reschedule_keeps_assignment(client):
    """改期失败（版本不符）整体回滚：会议室分配保持不变。"""
    _room(client, "room-1", 4)
    res1 = _reservation(client, "req-1", "2026-03-09T13:00:00Z",
                        "2026-03-09T14:00:00Z", ("alice",))
    _batch(client, "batch-1", [_item(res1["reservation_id"], 3)])
    resp = client.put(f"/api/v1/reservations/{res1['reservation_id']}", json={
        "expected_version": 99,
        "start_utc": "2026-03-09T15:00:00Z",
        "end_utc": "2026-03-09T16:00:00Z",
        "participant_ids": ["alice"],
    })
    assert resp.status_code == 409
    current = _assignments(client)
    assert [(a["reservation_id"], a["room_id"])
            for a in current] == [(res1["reservation_id"], "room-1")]


# ------------------------------------------------------------------ 并发

def test_concurrent_batches_single_winner(db_path):
    """并发批次竞争同一房间：BEGIN IMMEDIATE 串行化，恰一个赢家。"""
    from app import reservations as reservation_core

    room_core.upsert_room("room-1", 4, (), db_path=db_path)
    r1, _ = reservation_core.create_reservation(
        request_id="req-1", start_utc=_utc(2026, 3, 9, 13),
        end_utc=_utc(2026, 3, 9, 14), participant_ids=["alice"],
        db_path=db_path)
    r2, _ = reservation_core.create_reservation(
        request_id="req-2", start_utc=_utc(2026, 3, 9, 13),
        end_utc=_utc(2026, 3, 9, 14), participant_ids=["bob"],
        db_path=db_path)

    barrier = threading.Barrier(2)
    won, lost = [], []

    def worker(batch_id, reservation_id):
        barrier.wait()
        try:
            won.append(room_core.assign_batch(
                batch_id, [(reservation_id, 3, ())], db_path=db_path))
        except room_core.BatchUnsolvableError as exc:
            lost.append(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        pool.submit(worker, "batch-1", r1.id)
        pool.submit(worker, "batch-2", r2.id)

    assert len(won) == 1
    assert len(lost) == 1
    # 输家看到的就是赢家的那条预约无法分配之外的另一条
    assert len(room_core.list_assignments(db_path=db_path)) == 1


def test_concurrent_idempotent_replays_single_batch(db_path):
    """同一 batch_id + 同一内容并发重试：只新建一个批次，结果一致。"""
    from app import reservations as reservation_core

    room_core.upsert_room("room-1", 4, (), db_path=db_path)
    r1, _ = reservation_core.create_reservation(
        request_id="req-1", start_utc=_utc(2026, 3, 9, 13),
        end_utc=_utc(2026, 3, 9, 14), participant_ids=["alice"],
        db_path=db_path)

    barrier = threading.Barrier(8)
    results = []

    def worker(_):
        barrier.wait()
        results.append(room_core.assign_batch(
            "batch-1", [(r1.id, 3, ())], db_path=db_path))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(8)))

    assert len(results) == 8
    assert sum(1 for _, created in results if created) == 1
    batches = {result.batch_id for result, _ in results}
    assert batches == {"batch-1"}
    assert len(room_core.list_assignments(db_path=db_path)) == 1


# ------------------------------------------------------------------ 迁移

def test_init_creates_room_tables_losslessly(tmp_path, monkeypatch):
    """旧版库（无会议室相关表）初始化后新增三表，既有数据不动。"""
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
        assert {"meeting_rooms", "room_assignment_batches",
                "room_assignments"} <= names
        # 既有预约无损
        assert conn.execute(
            "SELECT request_id, participants_key FROM reservations"
        ).fetchone() == ("old", "[\"alice\"]")
    finally:
        conn.close()

    # 新表立即可用：在旧库上 upsert 会议室并完成一次批量分配
    room, created = room_core.upsert_room("room-1", 4, ["tv"], db_path=path)
    assert created is True
    result, created = room_core.assign_batch(
        "batch-1", [(1, 3, ["tv"])], db_path=path)
    assert created is True
    assert [(a.reservation_id, a.room_id)
            for a in result.assignments] == [(1, "room-1")]
