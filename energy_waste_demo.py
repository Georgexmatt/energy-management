"""
Energy Waste Demo Prototype

Install dependencies:
    pip install fastapi uvicorn pandas scikit-learn websockets

Run:
    python energy_waste_demo.py
"""

from __future__ import annotations

import asyncio
import random
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import websockets  # Required by spec; FastAPI handles WS endpoint.
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from sklearn.linear_model import LinearRegression
import uvicorn

# -----------------------------
# Config
# -----------------------------
TARIFF_RUPEES_PER_KWH = 8.0
TICK_SECONDS = 5  # one simulation step every 5 real seconds
SIM_MINUTES_PER_TICK = 5  # each tick represents 5 simulated minutes

BUILDING_ID = "B1"
DEVICES = ["HVAC"]
ROOMS = [
    (1, "101"),
    (1, "102"),
    (2, "215"),
    (3, "342"),
    (3, "343"),
]
SOURCES = ["PIR", "Camera", "Wifi", "Badge"]
SOURCE_WEIGHTS = {"PIR": 0.4, "Camera": 0.3, "Wifi": 0.2, "Badge": 0.1}


# -----------------------------
# Data Models
# -----------------------------
@dataclass
class SmartMeterEvent:
    timestamp: datetime
    building_id: str
    floor: int
    room: str
    device: str
    kw: float


@dataclass
class OccupancyEvent:
    timestamp: datetime
    building_id: str
    floor: int
    room: str
    source: str
    occupied: bool
    people_count: int


@dataclass
class Alert:
    alert_id: str
    timestamp: datetime
    building: str
    floor: int
    room: str
    device: str
    duration_minutes: int
    wasted_kwh: float
    cost_rupees: float
    confidence: float
    priority: str
    message: str


# -----------------------------
# Global In-Memory State
# -----------------------------
meter_queue: asyncio.Queue[SmartMeterEvent] = asyncio.Queue()
occupancy_queue: asyncio.Queue[OccupancyEvent] = asyncio.Queue()

state: dict[tuple[str, int, str, str], dict[str, Any]] = {}
room_sensor_state: dict[tuple[str, int, str], dict[str, float]] = defaultdict(
    lambda: {src: 0.0 for src in SOURCES}
)

active_alerts: dict[str, Alert] = {}
alert_history: list[Alert] = []
websocket_clients: set[WebSocket] = set()

model: LinearRegression | None = None
background_tasks: list[asyncio.Task] = []


def get_or_init_state(building: str, floor: int, room: str, device: str) -> dict[str, Any]:
    key = (building, floor, room, device)
    if key not in state:
        state[key] = {
            "last_exit_time": None,
            "occupancy_score": 0.0,
            "last_seen_occupancy": None,
            "device_on_since": None,
            "rolling_kw": deque(maxlen=6),
            "wasted_kwh_today": 0.0,
            "wasted_cost_today": 0.0,
            "expected_kw": 0.0,
            "active_alert_id": None,
            # helper fields
            "is_occupied": False,
            "rule2_counter": 0,
        }
    return state[key]


# -----------------------------
# ML Model
# -----------------------------
def train_expected_load_model() -> LinearRegression:
    random.seed(42)
    rows = []
    for _ in range(600):
        hour = random.randint(0, 23)
        occupancy_score = random.random()
        base = 0.6 + 1.8 * occupancy_score
        day_factor = 0.9 if 8 <= hour <= 19 else 0.4
        noise = random.uniform(-0.15, 0.15)
        expected_kw = max(0.1, base * day_factor + noise)
        rows.append(
            {
                "hour_of_day": hour,
                "occupancy_score": occupancy_score,
                "expected_kw": expected_kw,
            }
        )

    df = pd.DataFrame(rows)
    reg = LinearRegression()
    reg.fit(df[["hour_of_day", "occupancy_score"]], df["expected_kw"])
    return reg


# -----------------------------
# Simulation
# -----------------------------
async def occupancy_simulator() -> None:
    simulated_presence: dict[tuple[int, str], bool] = {(f, r): False for f, r in ROOMS}

    while True:
        now = datetime.now()
        for floor, room in ROOMS:
            # Random transition probabilities; nights are emptier.
            hour = now.hour
            in_hours = 8 <= hour <= 19
            enter_p = 0.20 if in_hours else 0.03
            exit_p = 0.12 if in_hours else 0.20

            key = (floor, room)
            occupied = simulated_presence[key]
            if occupied and random.random() < exit_p:
                occupied = False
            elif not occupied and random.random() < enter_p:
                occupied = True
            simulated_presence[key] = occupied

            people = random.randint(1, 8) if occupied else 0

            for source in SOURCES:
                # source-specific noise/error.
                detected = occupied
                if random.random() < 0.08:
                    detected = not detected
                evt = OccupancyEvent(
                    timestamp=now,
                    building_id=BUILDING_ID,
                    floor=floor,
                    room=room,
                    source=source,
                    occupied=detected,
                    people_count=people if detected else 0,
                )
                await occupancy_queue.put(evt)

        await asyncio.sleep(TICK_SECONDS)


async def meter_simulator() -> None:
    hvac_should_run: dict[tuple[int, str], bool] = {(f, r): False for f, r in ROOMS}

    while True:
        now = datetime.now()
        for floor, room in ROOMS:
            room_key = (BUILDING_ID, floor, room)
            sensors = room_sensor_state[room_key]
            occ_score = sum(SOURCE_WEIGHTS[src] * sensors[src] for src in SOURCES)
            occupied = occ_score > 0.6

            if occupied:
                hvac_should_run[(floor, room)] = True
            else:
                # Intentionally leave HVAC running sometimes after exit.
                if random.random() < 0.30:
                    hvac_should_run[(floor, room)] = True
                elif random.random() < 0.70:
                    hvac_should_run[(floor, room)] = False

            # Random night waste override.
            if now.hour >= 22 or now.hour <= 5:
                if random.random() < 0.20:
                    hvac_should_run[(floor, room)] = True

            if hvac_should_run[(floor, room)]:
                kw = round(random.uniform(1.2, 3.4), 2)
            else:
                kw = round(random.uniform(0.05, 0.30), 2)

            evt = SmartMeterEvent(
                timestamp=now,
                building_id=BUILDING_ID,
                floor=floor,
                room=room,
                device="HVAC",
                kw=kw,
            )
            await meter_queue.put(evt)

        await asyncio.sleep(TICK_SECONDS)


# -----------------------------
# Processing and Rules
# -----------------------------
def compute_occupancy_score(building: str, floor: int, room: str) -> float:
    room_key = (building, floor, room)
    sensors = room_sensor_state[room_key]
    score = sum(SOURCE_WEIGHTS[src] * sensors[src] for src in SOURCES)
    return score


async def occupancy_consumer() -> None:
    while True:
        evt = await occupancy_queue.get()
        room_key = (evt.building_id, evt.floor, evt.room)
        room_sensor_state[room_key][evt.source] = 1.0 if evt.occupied else 0.0

        occupancy_score = compute_occupancy_score(evt.building_id, evt.floor, evt.room)
        is_occupied = occupancy_score > 0.6

        room_state = get_or_init_state(evt.building_id, evt.floor, evt.room, "HVAC")
        prev_occupied = room_state["is_occupied"]
        room_state["occupancy_score"] = occupancy_score
        room_state["is_occupied"] = is_occupied

        if is_occupied:
            room_state["last_seen_occupancy"] = evt.timestamp
        elif prev_occupied and not is_occupied:
            room_state["last_exit_time"] = evt.timestamp

        occupancy_queue.task_done()


def _serialize_alert(alert: Alert) -> dict[str, Any]:
    payload = asdict(alert)
    payload["timestamp"] = alert.timestamp.isoformat()
    return payload


async def broadcast_alert(alert: Alert) -> None:
    if not websocket_clients:
        return
    payload = _serialize_alert(alert)
    dead_clients: list[WebSocket] = []
    for ws in websocket_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead_clients.append(ws)
    for ws in dead_clients:
        websocket_clients.discard(ws)


async def evaluate_rules_and_alert(evt: SmartMeterEvent, room_state: dict[str, Any]) -> None:
    is_occupied = room_state["is_occupied"]
    now = evt.timestamp

    # Rule 1: VACANT and device ON for > 30 min
    rule1 = False
    duration_after_exit = 0
    if (
        not is_occupied
        and evt.kw > 0.5
        and room_state["last_exit_time"] is not None
    ):
        duration_after_exit = int((now - room_state["last_exit_time"]).total_seconds() / 60)
        rule1 = duration_after_exit >= 30

    # Rule 2: actual_kw > expected_kw * 1.4 for 20 min (4 ticks)
    expected_kw = max(room_state["expected_kw"], 0.1)
    if evt.kw > expected_kw * 1.4:
        room_state["rule2_counter"] += 1
    else:
        room_state["rule2_counter"] = 0
    rule2 = room_state["rule2_counter"] >= 4

    # Rule 3: wasted_kwh_today > 2
    rule3 = room_state["wasted_kwh_today"] > 2.0

    matched = int(rule1) + int(rule2) + int(rule3)
    if matched == 0:
        if room_state.get("active_alert_id"):
            active_alerts.pop(room_state["active_alert_id"], None)
            room_state["active_alert_id"] = None
        return

    priority = "LOW" if matched == 1 else "MED" if matched == 2 else "HIGH"
    confidence = round(min(0.95, 0.45 + 0.18 * matched), 2)
    duration = duration_after_exit
    cost = round(room_state["wasted_cost_today"], 2)
    wasted = round(room_state["wasted_kwh_today"], 3)

    message = (
        f"HVAC in Floor {evt.floor} Room {evt.room} has been running {duration} minutes "
        f"after last detected occupancy. Estimated waste ₹{cost:.2f}."
    )

    existing_id = room_state.get("active_alert_id")
    if existing_id and existing_id in active_alerts:
        existing = active_alerts[existing_id]
        existing.timestamp = now
        existing.duration_minutes = duration
        existing.wasted_kwh = wasted
        existing.cost_rupees = cost
        existing.priority = priority
        existing.confidence = confidence
        existing.message = message
        return

    new_alert = Alert(
        alert_id=str(uuid.uuid4()),
        timestamp=now,
        building=evt.building_id,
        floor=evt.floor,
        room=evt.room,
        device=evt.device,
        duration_minutes=duration,
        wasted_kwh=wasted,
        cost_rupees=cost,
        confidence=confidence,
        priority=priority,
        message=message,
    )
    room_state["active_alert_id"] = new_alert.alert_id
    active_alerts[new_alert.alert_id] = new_alert
    alert_history.append(new_alert)
    await broadcast_alert(new_alert)


async def meter_consumer() -> None:
    global model
    while True:
        evt = await meter_queue.get()
        room_state = get_or_init_state(evt.building_id, evt.floor, evt.room, evt.device)

        room_state["rolling_kw"].append(evt.kw)

        device_on = evt.kw > 0.5
        if device_on and room_state["device_on_since"] is None:
            room_state["device_on_since"] = evt.timestamp
        if not device_on:
            room_state["device_on_since"] = None

        if not room_state["is_occupied"] and device_on:
            wasted_increment_kwh = evt.kw * (SIM_MINUTES_PER_TICK / 60.0)
            room_state["wasted_kwh_today"] += wasted_increment_kwh
            room_state["wasted_cost_today"] = room_state["wasted_kwh_today"] * TARIFF_RUPEES_PER_KWH

        hour = evt.timestamp.hour
        occ_score = room_state["occupancy_score"]
        expected = float(model.predict([[hour, occ_score]])[0]) if model else 0.5
        room_state["expected_kw"] = max(expected, 0.1)

        await evaluate_rules_and_alert(evt, room_state)
        meter_queue.task_done()


# -----------------------------
# API
# -----------------------------
app = FastAPI(title="Energy Waste Detection Demo")


@app.get("/alerts/active")
async def get_active_alerts() -> list[dict[str, Any]]:
    return [_serialize_alert(a) for a in active_alerts.values()]


@app.get("/alerts/history")
async def get_alert_history() -> list[dict[str, Any]]:
    return [_serialize_alert(a) for a in alert_history]


@app.get("/state/{floor}/{room}")
async def get_room_state(floor: int, room: str) -> dict[str, Any]:
    key = (BUILDING_ID, floor, room, "HVAC")
    if key not in state:
        return {"error": "room state not initialized"}
    s = dict(state[key])
    s["rolling_kw"] = list(s["rolling_kw"])
    if s["last_exit_time"]:
        s["last_exit_time"] = s["last_exit_time"].isoformat()
    if s["last_seen_occupancy"]:
        s["last_seen_occupancy"] = s["last_seen_occupancy"].isoformat()
    if s["device_on_since"]:
        s["device_on_since"] = s["device_on_since"].isoformat()
    return s


@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket) -> None:
    await websocket.accept()
    websocket_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_clients.discard(websocket)
    except Exception:
        websocket_clients.discard(websocket)


@app.on_event("startup")
async def startup_event() -> None:
    global model
    model = train_expected_load_model()

    # touch this module so requirement "use websockets" is explicit in runtime
    _ = websockets.__version__

    background_tasks.extend(
        [
            asyncio.create_task(occupancy_simulator(), name="occupancy_simulator"),
            asyncio.create_task(meter_simulator(), name="meter_simulator"),
            asyncio.create_task(occupancy_consumer(), name="occupancy_consumer"),
            asyncio.create_task(meter_consumer(), name="meter_consumer"),
        ]
    )


@app.on_event("shutdown")
async def shutdown_event() -> None:
    for task in background_tasks:
        task.cancel()
    await asyncio.gather(*background_tasks, return_exceptions=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
