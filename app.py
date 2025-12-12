from __future__ import annotations
"""
OmegaV4 – Unified Autonomous Oversight Kernel (deterministic, inspectable, human-gated)

Run with:
  streamlit run app.py

OmegaV4 upgrades over V3 (high-signal, ops-grade):
1) Counterfactual risk attribution (Δ-risk): shows what drove risk this tick.
2) Epistemic uncertainty + "knowledge hole" detection: can force HOLD / human gate.
3) Invariant proof objects: margins + confidence + evidence hashes per invariant.
4) Gate load regulator: gate pressure index + cooldown + supervision-withdraw recommendation.
5) Policy shadow diff: run alternate policies in parallel and compare bands / gates.
6) Deterministic replay capsule: one-file export/import of state + audit + memory.

Omega never actuates; it only observes, analyzes, and proposes.
Humans remain the final authority.
"""
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Protocol, Tuple

import pandas as pd
import streamlit as st

# ============================================================================
# Utilities: hashing, time, deterministic RNG
# ============================================================================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def tick_rng(seed: int, tick: int, salt: int = 0) -> random.Random:
    """
    Deterministic per-tick RNG derived from (seed, tick, salt).
    Keeps all randomness local and replayable given the same seed + tick.
    """
    mixed = (seed ^ (tick * 0x9E3779B9) ^ (salt * 0x85EBCA6B)) & 0xFFFFFFFF
    return random.Random(mixed)


# ============================================================================
# Audit spine – tamper-evident hash chain
# ============================================================================

@dataclass
class AuditEntry:
    seq: int
    timestamp: str
    kind: str
    payload: Dict[str, Any]
    prev_hash: str
    hash: str
    session_id: str


class AuditSpine:
    """
    Tamper-evident hash chain for all events.

    - Append-only
    - Each entry includes previous hash
    - Hash is over (serialized entry + prev_hash)
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id = session_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        self.genesis_hash: str = sha256_json({"genesis": self.session_id})
        self.prev_hash: str = self.genesis_hash
        self.entries: List[AuditEntry] = []
        self.seq: int = 0

    def log(self, kind: str, payload: Dict[str, Any]) -> AuditEntry:
        self.seq += 1
        ts = utc_now_iso()
        body = {
            "session_id": self.session_id,
            "seq": self.seq,
            "timestamp": ts,
            "kind": kind,
            "payload": payload,
            "prev_hash": self.prev_hash,
        }
        h = sha256_bytes((json.dumps(body, sort_keys=True) + self.prev_hash).encode("utf-8"))
        body["hash"] = h
        entry = AuditEntry(
            seq=self.seq,
            timestamp=ts,
            kind=kind,
            payload=payload,
            prev_hash=self.prev_hash,
            hash=h,
            session_id=self.session_id,
        )
        self.prev_hash = h
        self.entries.append(entry)
        return entry

    def to_json(self) -> str:
        serializable = [
            {
                "session_id": e.session_id,
                "seq": e.seq,
                "timestamp": e.timestamp,
                "kind": e.kind,
                "payload": e.payload,
                "prev_hash": e.prev_hash,
                "hash": e.hash,
            }
            for e in self.entries
        ]
        return json.dumps(serializable, indent=2, sort_keys=True, ensure_ascii=False)

    def load_from_json(self, raw: str) -> None:
        """
        Load an existing audit chain (for replay capsule import).
        The next appended entry will extend the loaded chain.
        """
        data = json.loads(raw)
        entries: List[AuditEntry] = []
        for e in data:
            entries.append(
                AuditEntry(
                    seq=int(e["seq"]),
                    timestamp=str(e["timestamp"]),
                    kind=str(e["kind"]),
                    payload=dict(e["payload"]),
                    prev_hash=str(e["prev_hash"]),
                    hash=str(e["hash"]),
                    session_id=str(e["session_id"]),
                )
            )
        if entries:
            self.session_id = entries[0].session_id
            self.genesis_hash = sha256_json({"genesis": self.session_id})
            self.entries = entries
            self.seq = entries[-1].seq
            self.prev_hash = entries[-1].hash
        else:
            # Empty chain: keep current genesis
            self.entries = []
            self.seq = 0
            self.prev_hash = self.genesis_hash

    def tail(self, n: int = 32) -> List[AuditEntry]:
        return self.entries[-n:]

    def verify_chain(self) -> Tuple[bool, Optional[int]]:
        """
        Verify the audit chain. Returns (ok, first_bad_index or None).
        """
        prev = self.genesis_hash
        for idx, e in enumerate(self.entries):
            body = {
                "session_id": e.session_id,
                "seq": e.seq,
                "timestamp": e.timestamp,
                "kind": e.kind,
                "payload": e.payload,
                "prev_hash": prev,
            }
            expected_hash = sha256_bytes((json.dumps(body, sort_keys=True) + prev).encode("utf-8"))
            if e.prev_hash != prev or e.hash != expected_hash:
                return False, idx
            prev = e.hash
        return True, None


# ============================================================================
# Asset identity, world model, faults, sensors
# ============================================================================

@dataclass(frozen=True)
class AssetId:
    site: str
    asset: str

    def key(self) -> str:
        return f"{self.site}:{self.asset}"


class GovernorMode:
    SHADOW = "shadow"
    TRAINING = "training"
    LIVE = "live"


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    heading_deg: float = 0.0
    zone: str = "zone-A"


@dataclass
class FaultProfile:
    drift_spike_prob: float = 0.0
    dropout_prob: float = 0.0
    freeze_prob: float = 0.0
    negative_speed_prob: float = 0.0
    contradictory_prob: float = 0.0
    timestamp_anomaly_prob: float = 0.0


@dataclass
class SensorChannel:
    name: str
    value: float
    last_true_value: float
    last_update_tick: int
    ok: bool = True
    stale: bool = False
    frozen: bool = False
    contradictory: bool = False
    future_timestamp: bool = False


@dataclass
class SensorSuite:
    drift: List[SensorChannel]
    speed: List[SensorChannel]
    stability: List[SensorChannel]


def default_sensor_suite() -> SensorSuite:
    return SensorSuite(
        drift=[
            SensorChannel("drift_imu", 0.0, 0.0, 0),
            SensorChannel("drift_camera", 0.0, 0.0, 0),
            SensorChannel("drift_fused", 0.0, 0.0, 0),
        ],
        speed=[
            SensorChannel("speed_encoder", 40.0, 40.0, 0),
            SensorChannel("speed_gnss", 40.0, 40.0, 0),
        ],
        stability=[
            SensorChannel("stability_estimator", 100.0, 100.0, 0),
            SensorChannel("stability_redundant", 100.0, 100.0, 0),
        ],
    )


@dataclass
class VehicleState:
    drift_deg: float = 0.0
    stability: float = 100.0
    speed_kph: float = 40.0
    commanded_speed_kph: float = 40.0
    last_action: str = "none"


@dataclass
class WorldState:
    asset: AssetId
    tick: int = 0
    mode: str = GovernorMode.SHADOW
    vehicle: VehicleState = field(default_factory=VehicleState)
    pose: Pose = field(default_factory=Pose)
    sensors: SensorSuite = field(default_factory=default_sensor_suite)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def simulate_dynamics(world: WorldState, seed: int, faults: FaultProfile) -> WorldState:
    """
    Dynamics-only step (no sensor updates):

    - Actuator lag: speed approaches commanded_speed_kph
    - Drift and stability random walk
    - Pose propagation along x-axis
    - Faults: drift spikes, negative speed
    """
    rng = tick_rng(seed, world.tick, salt=1)
    v = world.vehicle

    # Random walk commanded speed (e.g., operator/autopilot commands)
    cmd_delta = rng.uniform(-2.0, 2.0)
    v.commanded_speed_kph = clamp(v.commanded_speed_kph + cmd_delta, 0.0, 80.0)

    # Actuator lag: speed approaches commanded with limited rate
    lag_limit = 5.0
    diff = v.commanded_speed_kph - v.speed_kph
    diff = clamp(diff, -lag_limit, lag_limit)
    v.speed_kph = clamp(v.speed_kph + diff, -40.0, 80.0)

    # Base drift and stability random walk
    v.drift_deg = clamp(v.drift_deg + rng.uniform(-1.5, 1.5), -90.0, 90.0)
    v.stability = clamp(v.stability + rng.uniform(-3.0, 2.0), 0.0, 100.0)

    # Faults on true state
    if rng.random() < faults.drift_spike_prob:
        v.drift_deg = clamp(v.drift_deg + rng.uniform(-20.0, 20.0), -90.0, 90.0)
    if rng.random() < faults.negative_speed_prob:
        v.speed_kph = -abs(v.speed_kph)

    # Pose propagation (1D along x for simplicity)
    dt = 1.0  # abstract time units
    speed_mps = v.speed_kph / 3.6
    world.pose.x += speed_mps * dt

    # Simple zone mapping: zones alternate every 200 units
    zone_cycle = abs(world.pose.x) % 200.0
    world.pose.zone = "zone-A" if zone_cycle < 100.0 else "zone-B"
    return world


def update_sensors(world: WorldState, seed: int, faults: FaultProfile) -> None:
    """
    Update sensor readings based on true state and fault profile.

    Models:
      - noise
      - dropout
      - freeze
      - contradictions
      - timestamp anomalies
    """
    rng = tick_rng(seed, world.tick, salt=2)

    def update_channel(ch: SensorChannel, true_val: float, noise_span: float) -> None:
        ch.future_timestamp = False

        # If frozen, we keep the previous value and let staleness grow
        if ch.frozen:
            # Freeze can thaw with small probability
            if rng.random() < 0.05:
                ch.frozen = False
            else:
                # Staleness computed below
                ch.stale = (world.tick - ch.last_update_tick) > 3
                return

        # Dropout: no update (ok=False)
        if rng.random() < faults.dropout_prob:
            ch.ok = False
            # Keep last value, but still update stale flag
            ch.stale = (world.tick - ch.last_update_tick) > 3
            return

        # Normal update
        ch.ok = True
        noisy = true_val + rng.uniform(-noise_span, noise_span)
        ch.value = noisy
        ch.last_true_value = true_val
        ch.last_update_tick = world.tick

        # Contradictory reading
        if rng.random() < faults.contradictory_prob:
            ch.value = noisy + rng.uniform(20.0, 40.0) * (1.0 if rng.random() < 0.5 else -1.0)
            ch.contradictory = True
        else:
            ch.contradictory = False

        # Freeze the channel
        if rng.random() < faults.freeze_prob:
            ch.frozen = True

        # Timestamp anomalies: move timestamp forward/backward by a few ticks
        if rng.random() < faults.timestamp_anomaly_prob:
            offset = rng.randint(3, 10)
            if rng.random() < 0.5:
                ch.last_update_tick = world.tick - offset
            else:
                ch.last_update_tick = world.tick + offset

        ch.future_timestamp = ch.last_update_tick > world.tick
        ch.stale = (world.tick - ch.last_update_tick) > 3

    v = world.vehicle
    for ch in world.sensors.drift:
        update_channel(ch, v.drift_deg, noise_span=1.5)
    for ch in world.sensors.speed:
        update_channel(ch, v.speed_kph, noise_span=2.5)
    for ch in world.sensors.stability:
        update_channel(ch, v.stability, noise_span=5.0)


# ============================================================================
# Multi-asset interactions helpers
# ============================================================================

MAX_PROX_DISTANCE = 200.0
MIN_CONVOY_SPACING = 20.0
ZONE_CONFLICT_THRESHOLD = 10.0


@dataclass
class InteractionSummary:
    nearest_distance: float
    same_zone_conflict: bool
    convoy_spacing_bad: bool


def compute_interactions(worlds: Dict[str, WorldState]) -> Dict[str, InteractionSummary]:
    result: Dict[str, InteractionSummary] = {}
    keys = list(worlds.keys())
    for key in keys:
        w = worlds[key]
        x = w.pose.x
        zone = w.pose.zone
        nearest = float("inf")
        same_zone_conflict = False

        for other_key in keys:
            if other_key == key:
                continue
            w2 = worlds[other_key]
            if w2.asset.site != w.asset.site:
                continue
            dx = abs(w2.pose.x - x)
            if dx < nearest:
                nearest = dx
            if w2.pose.zone == zone and dx < ZONE_CONFLICT_THRESHOLD:
                same_zone_conflict = True

        if nearest == float("inf"):
            nearest = MAX_PROX_DISTANCE

        convoy_spacing_bad = nearest < MIN_CONVOY_SPACING
        result[key] = InteractionSummary(
            nearest_distance=nearest,
            same_zone_conflict=same_zone_conflict,
            convoy_spacing_bad=convoy_spacing_bad,
        )
    return result


# ============================================================================
# Sensor metrics + epistemic uncertainty (knowledge holes)
# ============================================================================

@dataclass
class SensorMetrics:
    sensor_health: float
    stale_fraction: float
    contradictory_fraction: float
    dropout_fraction: float
    future_timestamp_fraction: float
    frozen_fraction: float


def compute_sensor_metrics(world: WorldState) -> SensorMetrics:
    channels = world.sensors.drift + world.sensors.speed + world.sensors.stability
    if not channels:
        return SensorMetrics(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    total = len(channels)
    stale_count = sum(1 for c in channels if c.stale)
    contradictory_count = sum(1 for c in channels if c.contradictory)
    dropout_count = sum(1 for c in channels if not c.ok)
    future_ts_count = sum(1 for c in channels if c.future_timestamp)
    frozen_count = sum(1 for c in channels if c.frozen)

    bad_count = sum(1 for c in channels if (not c.ok) or c.stale or c.contradictory or c.future_timestamp)
    sensor_health = max(0.0, 1.0 - bad_count / float(total))

    return SensorMetrics(
        sensor_health=sensor_health,
        stale_fraction=stale_count / float(total),
        contradictory_fraction=contradictory_count / float(total),
        dropout_fraction=dropout_count / float(total),
        future_timestamp_fraction=future_ts_count / float(total),
        frozen_fraction=frozen_count / float(total),
    )


@dataclass
class EpistemicConfig:
    knowledge_hole_threshold: float = 0.60


@dataclass
class EpistemicPacket:
    tick: int
    epistemic: float  # 0–1 (higher = more unknown)
    knowledge_hole: bool
    signals: Dict[str, float]


def compute_epistemic(
    tick: int,
    metrics: SensorMetrics,
    drift_spread: float,
    speed_spread: float,
    stability_spread: float,
    cfg: EpistemicConfig,
) -> EpistemicPacket:
    # Normalize spreads (tuned to rough expected ranges)
    drift_spread_norm = clamp(drift_spread / 10.0, 0.0, 1.0)
    speed_spread_norm = clamp(speed_spread / 15.0, 0.0, 1.0)
    stability_spread_norm = clamp(stability_spread / 30.0, 0.0, 1.0)
    spread_norm = (drift_spread_norm * 0.34 + speed_spread_norm * 0.33 + stability_spread_norm * 0.33)

    # Weighted epistemic uncertainty score
    epistemic = 0.0
    epistemic += (1.0 - metrics.sensor_health) * 0.35
    epistemic += metrics.stale_fraction * 0.15
    epistemic += metrics.contradictory_fraction * 0.25
    epistemic += metrics.dropout_fraction * 0.10
    epistemic += metrics.future_timestamp_fraction * 0.10
    epistemic += spread_norm * 0.15

    epistemic = clamp(epistemic, 0.0, 1.0)
    knowledge_hole = epistemic >= cfg.knowledge_hole_threshold

    signals = {
        "sensor_health": round(metrics.sensor_health, 3),
        "stale_fraction": round(metrics.stale_fraction, 3),
        "contradictory_fraction": round(metrics.contradictory_fraction, 3),
        "dropout_fraction": round(metrics.dropout_fraction, 3),
        "future_timestamp_fraction": round(metrics.future_timestamp_fraction, 3),
        "frozen_fraction": round(metrics.frozen_fraction, 3),
        "drift_spread_norm": round(drift_spread_norm, 3),
        "speed_spread_norm": round(speed_spread_norm, 3),
        "stability_spread_norm": round(stability_spread_norm, 3),
        "spread_norm": round(spread_norm, 3),
    }

    return EpistemicPacket(tick=tick, epistemic=epistemic, knowledge_hole=knowledge_hole, signals=signals)


# ============================================================================
# Safety kernel – contextual risk + hysteresis + attribution (Δ-risk)
# ============================================================================

@dataclass
class RiskConfig:
    drift_watch_deg: float = 10.0
    drift_hold_deg: float = 25.0
    drift_stop_deg: float = 45.0
    stability_watch_min: float = 70.0
    stability_hold_min: float = 55.0
    stability_stop_min: float = 40.0


@dataclass
class RiskContext:
    sensor_health: float
    stale_fraction: float
    contradictory_fraction: float
    proximity: float
    same_zone_conflict: bool


@dataclass
class RiskState:
    last_band: str = "LOW"
    stable_ticks_in_band: int = 0
    history_risk_sum: float = 0.0
    history_risk_count: int = 0
    safety_credit: float = 0.0
    knowledge_hole_streak: int = 0


@dataclass
class RiskAttribution:
    contributions: Dict[str, float]              # per-factor contribution to risk (0–1 domain)
    counterfactual_deltas: Dict[str, float]      # Δ-risk if factor neutralized (positive => drove risk up)
    top_factors: List[Dict[str, Any]]            # sorted factors with delta magnitude


@dataclass
class RiskPacket:
    tick: int
    risk: float  # 0–1
    band: str    # hysteretic band: "LOW" | "WATCH" | "HOLD" | "STOP"
    features: Dict[str, float]
    attribution: RiskAttribution


class SafetyKernel:
    """
    Deterministic, contextual, hysteretic risk model.

    Inputs:
      - world (true state + sensors)
      - risk context (sensor health, proximity, conflicts)
      - risk state (history / last band / safety credit)

    Outputs:
      - RiskPacket with band + feature contributions + counterfactual deltas
    """

    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg
        self._severity_rank = {"LOW": 0, "WATCH": 1, "HOLD": 2, "STOP": 3}

    def _fuse_channels(self, channels: List[SensorChannel], default: float) -> Tuple[float, float, int]:
        usable = [c.value for c in channels if c.ok and not c.stale and (not c.future_timestamp)]
        if not usable:
            return default, 0.0, 0
        usable_sorted = sorted(usable)
        median = usable_sorted[len(usable_sorted) // 2]
        spread = (max(usable_sorted) - min(usable_sorted)) if len(usable_sorted) > 1 else 0.0
        return median, spread, len(usable)

    def _compute_risk_scalar(
        self,
        drift_norm: float,
        stability_norm: float,
        speed_norm: float,
        sensor_health: float,
        stale_fraction: float,
        contradictory_fraction: float,
        proximity: float,
        same_zone_conflict: bool,
        state: RiskState,
        include_history: bool = True,
        include_credit: bool = True,
    ) -> Tuple[float, Dict[str, float]]:
        # Core weights (sum to 1.0)
        drift_c = 0.45 * drift_norm
        stability_c = 0.35 * (1.0 - stability_norm)
        speed_c = 0.20 * speed_norm

        # Context weights
        sensor_c = (1.0 - sensor_health) * 0.25
        proximity_c = proximity * 0.30
        zone_c = 0.15 if same_zone_conflict else 0.0
        stale_c = stale_fraction * 0.10
        contradictory_c = contradictory_fraction * 0.15

        risk = drift_c + stability_c + speed_c + sensor_c + proximity_c + zone_c + stale_c + contradictory_c

        # Safety credit/debt (debt increases risk)
        debt_norm = 0.0
        credit_c = 0.0
        if include_credit:
            debt_norm = clamp(max(0.0, -state.safety_credit) / 50.0, 0.0, 1.0)
            credit_c = debt_norm * 0.15
            risk += credit_c

        # History bias: conservative if recent average risk > current estimate
        history_c = 0.0
        if include_history and state.history_risk_count > 0:
            avg_hist = state.history_risk_sum / state.history_risk_count
            if avg_hist > risk:
                history_c = 0.15 * (avg_hist - risk)
                risk += history_c

        risk = clamp(risk, 0.0, 1.0)

        contrib = {
            "drift": drift_c,
            "stability": stability_c,
            "speed": speed_c,
            "sensor_health": sensor_c,
            "proximity": proximity_c,
            "zone_conflict": zone_c,
            "stale": stale_c,
            "contradictory": contradictory_c,
            "credit_debt": credit_c,
            "history_bias": history_c,
        }
        return risk, contrib

    def eval(self, world: WorldState, ctx: RiskContext, state: RiskState) -> Tuple[RiskPacket, float, float, float]:
        v = world.vehicle
        sensors = world.sensors

        # Sensor fusion
        fused_drift, drift_spread, _ = self._fuse_channels(sensors.drift, v.drift_deg)
        fused_speed, speed_spread, _ = self._fuse_channels(sensors.speed, v.speed_kph)
        fused_stability, stability_spread, _ = self._fuse_channels(sensors.stability, v.stability)

        drift = abs(fused_drift)
        stability = fused_stability
        speed = fused_speed

        # Normalize core features
        drift_norm = min(1.0, drift / self.cfg.drift_stop_deg)

        # stability_norm ~ 1 when healthy, 0 when at/under stop min
        stability_norm = 1.0 - min(
            1.0,
            max(0.0, (self.cfg.stability_stop_min - stability) / 60.0),
        )

        speed_norm = min(1.0, max(0.0, speed) / 120.0)

        # Risk scalar + contributions
        risk, contrib = self._compute_risk_scalar(
            drift_norm=drift_norm,
            stability_norm=stability_norm,
            speed_norm=speed_norm,
            sensor_health=ctx.sensor_health,
            stale_fraction=ctx.stale_fraction,
            contradictory_fraction=ctx.contradictory_fraction,
            proximity=ctx.proximity,
            same_zone_conflict=ctx.same_zone_conflict,
            state=state,
            include_history=True,
            include_credit=True,
        )

        # Counterfactual deltas (neutralize one factor at a time)
        def cf(**overrides: Any) -> float:
            return self._compute_risk_scalar(
                drift_norm=float(overrides.get("drift_norm", drift_norm)),
                stability_norm=float(overrides.get("stability_norm", stability_norm)),
                speed_norm=float(overrides.get("speed_norm", speed_norm)),
                sensor_health=float(overrides.get("sensor_health", ctx.sensor_health)),
                stale_fraction=float(overrides.get("stale_fraction", ctx.stale_fraction)),
                contradictory_fraction=float(overrides.get("contradictory_fraction", ctx.contradictory_fraction)),
                proximity=float(overrides.get("proximity", ctx.proximity)),
                same_zone_conflict=bool(overrides.get("same_zone_conflict", ctx.same_zone_conflict)),
                state=state,
                include_history=True,
                include_credit=True,
            )[0]

        counterfactual_deltas = {
            "drift": round(risk - cf(drift_norm=0.0), 5),
            "stability": round(risk - cf(stability_norm=1.0), 5),
            "speed": round(risk - cf(speed_norm=0.0), 5),
            "sensor_health": round(risk - cf(sensor_health=1.0), 5),
            "proximity": round(risk - cf(proximity=0.0, same_zone_conflict=False), 5),
            "stale": round(risk - cf(stale_fraction=0.0), 5),
            "contradictory": round(risk - cf(contradictory_fraction=0.0), 5),
            "credit_debt": round(risk - self._compute_risk_scalar(
                drift_norm, stability_norm, speed_norm,
                ctx.sensor_health, ctx.stale_fraction, ctx.contradictory_fraction,
                ctx.proximity, ctx.same_zone_conflict,
                state, include_history=True, include_credit=False
            )[0], 5),
        }

        # Sort top factors by delta (descending)
        sorted_factors = sorted(counterfactual_deltas.items(), key=lambda kv: kv[1], reverse=True)
        top_factors = [{"factor": k, "delta": float(v)} for k, v in sorted_factors[:5]]

        # Update history
        state.history_risk_sum += risk
        state.history_risk_count += 1

        # Base band from thresholds
        if risk >= 0.80:
            base_band = "STOP"
        elif risk >= 0.55:
            base_band = "HOLD"
        elif risk >= 0.30:
            base_band = "WATCH"
        else:
            base_band = "LOW"

        old_band = state.last_band or "LOW"
        rank = self._severity_rank

        # Hysteresis: escalate immediately, de-escalate only after N stable ticks
        # V4: hysteresis is slightly affected by safety debt (more debt => slower de-escalation)
        debt_norm = clamp(max(0.0, -state.safety_credit) / 50.0, 0.0, 1.0)
        base_hyst = 3
        HYSTERESIS_TICKS = base_hyst + int(round(debt_norm * 2))

        if rank[base_band] > rank[old_band]:
            band = base_band
            state.stable_ticks_in_band = 0
        elif rank[base_band] < rank[old_band]:
            if state.stable_ticks_in_band >= HYSTERESIS_TICKS:
                band = base_band
                state.stable_ticks_in_band = 0
            else:
                band = old_band
                state.stable_ticks_in_band += 1
        else:
            band = base_band
            state.stable_ticks_in_band += 1

        state.last_band = band

        features: Dict[str, float] = {
            "drift_norm": round(drift_norm, 3),
            "stability_norm": round(stability_norm, 3),
            "speed_norm": round(speed_norm, 3),
            "sensor_health": round(ctx.sensor_health, 3),
            "stale_fraction": round(ctx.stale_fraction, 3),
            "contradictory_fraction": round(ctx.contradictory_fraction, 3),
            "proximity": round(ctx.proximity, 3),
            "drift_spread": round(drift_spread, 3),
            "speed_spread": round(speed_spread, 3),
            "stability_spread": round(stability_spread, 3),
            "safety_credit": round(state.safety_credit, 3),
            "safety_debt_norm": round(debt_norm, 3),
        }

        attribution = RiskAttribution(
            contributions={k: round(v, 5) for k, v in contrib.items()},
            counterfactual_deltas=counterfactual_deltas,
            top_factors=top_factors,
        )

        packet = RiskPacket(
            tick=world.tick,
            risk=risk,
            band=band,
            features=features,
            attribution=attribution,
        )
        return packet, drift_spread, speed_spread, stability_spread


# ============================================================================
# Governor – queue-based gating with stages and expiry + gate load regulator
# ============================================================================

@dataclass
class Invariants:
    drift_max_live: float
    stability_min_live: float
    max_tick_ms: float


@dataclass
class InvariantProof:
    invariant_id: str
    satisfied: bool
    margin: float
    confidence: float
    evidence_hashes: List[str]


@dataclass
class Decision:
    tick: int
    action: str  # "none" | "normal" | "cautious" | "stop_safe" | "hold_for_approval" | "withdraw_supervision"
    proposed_action: str
    band: str
    requires_human_gate: bool
    human_gate_id: Optional[str]
    invariants_violated: List[str]
    invariant_proofs: List[InvariantProof]
    reason_chain: List[Dict[str, Any]]
    gate_pressure_index: float
    knowledge_hole: bool
    epistemic: float
    safety_credit: float


@dataclass
class HumanGateState:
    gate_id: str
    asset_key: str
    tick: int
    band: str
    mode: str
    proposed_action: str
    severity: str
    stage: int = 1
    required_role: str = "supervisor"
    approved: Optional[bool] = None
    operator_id: Optional[str] = None
    note: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    expires_at_tick: int = 0
    resolved_at: Optional[str] = None


@dataclass
class SafetyPolicy:
    id: str
    risk_cfg: RiskConfig
    invariants: Invariants
    band_to_action: Dict[str, str]
    human_gate_bands_live: Tuple[str, ...]
    avalon_risk_cap: float
    source_hash: str


DEFAULT_POLICY_JSON_V4 = json.dumps(
    {
        "id": "policy_demo_v4",
        "risk_cfg": {
            "drift_watch_deg": 10.0,
            "drift_hold_deg": 25.0,
            "drift_stop_deg": 45.0,
            "stability_watch_min": 70.0,
            "stability_hold_min": 55.0,
            "stability_stop_min": 40.0,
        },
        "invariants": {
            "drift_max_live": 30.0,
            "stability_min_live": 60.0,
            "max_tick_ms": 50.0,
        },
        "band_to_action": {
            "LOW": "normal",
            "WATCH": "cautious",
            "HOLD": "stop_safe",
            "STOP": "stop_safe",
        },
        "human_gate_bands_live": ["HOLD", "STOP"],
        "avalon_risk_cap": 65.0,
    },
    sort_keys=True,
)


def _build_policy_from_dict(cfg: Dict[str, Any]) -> SafetyPolicy:
    required_top = {"id", "risk_cfg", "invariants", "band_to_action"}
    missing = required_top - set(cfg.keys())
    if missing:
        raise ValueError(f"Policy missing required fields: {sorted(missing)}")

    risk_cfg = RiskConfig(**cfg["risk_cfg"])
    invariants = Invariants(**cfg["invariants"])
    band_map = dict(cfg["band_to_action"])
    human_gate_bands = tuple(cfg.get("human_gate_bands_live", ["HOLD", "STOP"]))
    avalon_risk_cap = float(cfg.get("avalon_risk_cap", 65.0))
    src_hash = sha256_json(cfg)

    return SafetyPolicy(
        id=str(cfg.get("id", "policy_unnamed")),
        risk_cfg=risk_cfg,
        invariants=invariants,
        band_to_action=band_map,
        human_gate_bands_live=human_gate_bands,
        avalon_risk_cap=avalon_risk_cap,
        source_hash=src_hash,
    )


def load_default_policy() -> SafetyPolicy:
    return _build_policy_from_dict(json.loads(DEFAULT_POLICY_JSON_V4))


def build_shadow_policies(active: SafetyPolicy) -> Dict[str, SafetyPolicy]:
    """
    Create a couple of built-in shadow policies for diffing.

    - Conservative: earlier HOLD/STOP
    - Permissive: later HOLD/STOP
    """
    base = json.loads(DEFAULT_POLICY_JSON_V4)

    conservative = json.loads(DEFAULT_POLICY_JSON_V4)
    conservative["id"] = "policy_shadow_conservative"
    conservative["risk_cfg"]["drift_hold_deg"] = max(15.0, float(base["risk_cfg"]["drift_hold_deg"]) - 5.0)
    conservative["risk_cfg"]["drift_stop_deg"] = max(30.0, float(base["risk_cfg"]["drift_stop_deg"]) - 5.0)
    conservative["risk_cfg"]["stability_hold_min"] = min(80.0, float(base["risk_cfg"]["stability_hold_min"]) + 5.0)
    conservative["risk_cfg"]["stability_stop_min"] = min(70.0, float(base["risk_cfg"]["stability_stop_min"]) + 5.0)
    conservative["avalon_risk_cap"] = min(80.0, float(base.get("avalon_risk_cap", 65.0)) + 5.0)

    permissive = json.loads(DEFAULT_POLICY_JSON_V4)
    permissive["id"] = "policy_shadow_permissive"
    permissive["risk_cfg"]["drift_hold_deg"] = float(base["risk_cfg"]["drift_hold_deg"]) + 5.0
    permissive["risk_cfg"]["drift_stop_deg"] = float(base["risk_cfg"]["drift_stop_deg"]) + 5.0
    permissive["risk_cfg"]["stability_hold_min"] = max(30.0, float(base["risk_cfg"]["stability_hold_min"]) - 5.0)
    permissive["risk_cfg"]["stability_stop_min"] = max(20.0, float(base["risk_cfg"]["stability_stop_min"]) - 5.0)
    permissive["avalon_risk_cap"] = max(40.0, float(base.get("avalon_risk_cap", 65.0)) - 5.0)

    return {
        conservative["id"]: _build_policy_from_dict(conservative),
        permissive["id"]: _build_policy_from_dict(permissive),
    }


@dataclass
class GatePressureConfig:
    window_ticks: int = 50
    gates_per_window_soft: int = 3
    gates_per_window_hard: int = 6


class Governor:
    """
    Deterministic envelope governor:

    - maps risk band → proposed_action (via SafetyPolicy)
    - enforces mode semantics and invariants
    - maintains a queue of gates per asset (with expiry and stages)
    - V4: gate load regulator (pressure index, cooldown, supervision-withdraw recommendation)
    - V4: knowledge-hole gating (epistemic uncertainty)
    """

    def __init__(
        self,
        policy: SafetyPolicy,
        mode: str,
        asset_key: str,
        gate_pressure_cfg: Optional[GatePressureConfig] = None,
    ) -> None:
        self.policy = policy
        self.mode = mode
        self.asset_key = asset_key
        self.gate_pressure_cfg = gate_pressure_cfg or GatePressureConfig()
        self._next_gate_id = 1
        self._gates: Dict[str, HumanGateState] = {}
        self._severity_rank = {"LOW": 0, "WATCH": 1, "HOLD": 2, "STOP": 3}
        self._gate_created_ticks: List[int] = []
        self._last_gate_resolved_tick: Optional[int] = None

    def _new_gate(
        self,
        risk: RiskPacket,
        world: WorldState,
        proposed: str,
        stage: int = 1,
        expires_in_ticks: int = 5,
        reason: str = "",
    ) -> HumanGateState:
        gid = f"{self.asset_key}-gate-{self._next_gate_id:04d}"
        self._next_gate_id += 1
        severity = "critical" if risk.band == "STOP" else "high" if risk.band == "HOLD" else "elevated"
        required_role = "lead" if stage >= 2 else "supervisor"
        gate = HumanGateState(
            gate_id=gid,
            asset_key=self.asset_key,
            tick=world.tick,
            band=risk.band,
            mode=self.mode,
            proposed_action=proposed,
            severity=severity,
            stage=stage,
            required_role=required_role,
            expires_at_tick=world.tick + expires_in_ticks,
            note=reason.strip(),
        )
        self._gates[gid] = gate
        self._gate_created_ticks.append(world.tick)
        return gate

    def _update_gate_expirations(self, current_tick: int) -> None:
        """
        Expire and escalate gates that have not been resolved.
        """
        for gate in list(self._gates.values()):
            if gate.approved is None and current_tick > gate.expires_at_tick:
                # Auto-expire and escalate if not at max stage
                gate.approved = False
                gate.note = (gate.note or "") + " [auto-expired]"
                gate.resolved_at = utc_now_iso()
                self._last_gate_resolved_tick = current_tick

                if gate.stage < 3:
                    # Escalate
                    new_stage = gate.stage + 1
                    new_expires = 3 if new_stage >= 2 else 5
                    # Create escalation gate
                    dummy_world = WorldState(asset=AssetId("n/a", "n/a"), tick=current_tick)
                    self._new_gate(
                        risk=RiskPacket(
                            tick=current_tick,
                            risk=0.0,
                            band=gate.band,
                            features={},
                            attribution=RiskAttribution(contributions={}, counterfactual_deltas={}, top_factors=[]),
                        ),
                        world=dummy_world,
                        proposed=gate.proposed_action,
                        stage=new_stage,
                        expires_in_ticks=new_expires,
                        reason="escalated_after_expiry",
                    )

    def _active_gates(self) -> List[HumanGateState]:
        return [g for g in self._gates.values() if g.approved is None]

    def _select_current_gate(self) -> Optional[HumanGateState]:
        active = self._active_gates()
        if not active:
            return None

        def score(g: HumanGateState) -> Tuple[int, int, int]:
            return (
                self._severity_rank.get(g.band, 0),
                g.stage,
                g.tick,
            )

        return max(active, key=score)

    def gate_pressure_index(self, current_tick: int) -> float:
        cfg = self.gate_pressure_cfg
        window_start = current_tick - cfg.window_ticks
        recent = [t for t in self._gate_created_ticks if t >= window_start]
        recent_count = len(recent)
        # Normalize against soft/hard thresholds
        if recent_count <= cfg.gates_per_window_soft:
            return 0.0 if cfg.gates_per_window_soft == 0 else recent_count / float(cfg.gates_per_window_soft) * 0.5
        if recent_count >= cfg.gates_per_window_hard:
            return 1.0
        # Between soft and hard: map to (0.5..1.0)
        span = max(1, cfg.gates_per_window_hard - cfg.gates_per_window_soft)
        return clamp(0.5 + (recent_count - cfg.gates_per_window_soft) / float(span) * 0.5, 0.0, 1.0)

    def _gate_creation_cooldown(self, current_tick: int, pressure: float) -> int:
        # Base cooldown plus pressure-based extension
        base = 3
        extra = int(round(pressure * 6))
        return base + extra

    def _gate_creation_allowed(self, current_tick: int, pressure: float, band: str, knowledge_hole: bool) -> bool:
        # Always allow for STOP or knowledge holes (unknown is worse than risk here)
        if band == "STOP" or knowledge_hole:
            return True
        if self._last_gate_resolved_tick is None:
            return True
        cooldown = self._gate_creation_cooldown(current_tick, pressure)
        return (current_tick - self._last_gate_resolved_tick) >= cooldown

    def _invariant_proofs(
        self,
        world: WorldState,
        tick_ms: float,
        epistemic: float,
        mode: str,
    ) -> Tuple[List[InvariantProof], List[str]]:
        inv = self.policy.invariants
        v = world.vehicle
        proofs: List[InvariantProof] = []
        violated: List[str] = []

        # Confidence: epistemic low => high confidence, clamp to [0.05..0.99]
        confidence = clamp(1.0 - epistemic, 0.05, 0.99)

        # Drift invariant
        drift_margin = float(inv.drift_max_live - abs(v.drift_deg))
        drift_ok = drift_margin >= 0.0
        if mode == GovernorMode.LIVE and not drift_ok:
            violated.append("drift_exceeds_live_max")
        proofs.append(
            InvariantProof(
                invariant_id="drift_max_live",
                satisfied=(drift_ok if mode == GovernorMode.LIVE else True),
                margin=round(drift_margin, 3),
                confidence=round(confidence, 3),
                evidence_hashes=[sha256_json({"tick": world.tick, "drift_deg": v.drift_deg, "max": inv.drift_max_live})],
            )
        )

        # Stability invariant
        stability_margin = float(v.stability - inv.stability_min_live)
        stability_ok = stability_margin >= 0.0
        if mode == GovernorMode.LIVE and not stability_ok:
            violated.append("stability_below_live_min")
        proofs.append(
            InvariantProof(
                invariant_id="stability_min_live",
                satisfied=(stability_ok if mode == GovernorMode.LIVE else True),
                margin=round(stability_margin, 3),
                confidence=round(confidence, 3),
                evidence_hashes=[sha256_json({"tick": world.tick, "stability": v.stability, "min": inv.stability_min_live})],
            )
        )

        # Tick time invariant
        tick_margin = float(inv.max_tick_ms - tick_ms)
        tick_ok = tick_margin >= 0.0
        if mode == GovernorMode.LIVE and not tick_ok:
            violated.append("tick_overrun")
        proofs.append(
            InvariantProof(
                invariant_id="max_tick_ms",
                satisfied=(tick_ok if mode == GovernorMode.LIVE else True),
                margin=round(tick_margin, 3),
                confidence=round(confidence, 3),
                evidence_hashes=[sha256_json({"tick": world.tick, "tick_ms": tick_ms, "max": inv.max_tick_ms})],
            )
        )

        return proofs, violated

    def evaluate(
        self,
        risk: RiskPacket,
        world: WorldState,
        epistemic: EpistemicPacket,
        tick_ms: float,
    ) -> Decision:
        self._update_gate_expirations(world.tick)
        reasons: List[Dict[str, Any]] = []
        invariants_violated: List[str] = []

        band = risk.band
        proposed = self.policy.band_to_action.get(band, "stop_safe")

        reasons.append(
            {
                "rule": "band_to_action",
                "band": band,
                "proposed": proposed,
                "risk": round(risk.risk, 3),
            }
        )

        # Gate pressure
        pressure = self.gate_pressure_index(world.tick)
        reasons.append({"rule": "gate_pressure", "gate_pressure_index": round(pressure, 3)})

        # Knowledge holes override: treat unknown as requiring HOLD + gate (in non-shadow modes)
        knowledge_hole = epistemic.knowledge_hole
        if knowledge_hole:
            reasons.append(
                {
                    "rule": "knowledge_hole",
                    "epistemic": round(epistemic.epistemic, 3),
                    "note": "epistemic uncertainty above threshold",
                }
            )
            # Be conservative on proposals when epistemic is high
            proposed = "stop_safe" if self.mode == GovernorMode.LIVE else "hold_for_approval"

        # Invariant proofs (always produced; enforced only in LIVE)
        invariant_proofs, invariants_violated = self._invariant_proofs(
            world=world,
            tick_ms=tick_ms,
            epistemic=epistemic.epistemic,
            mode=self.mode,
        )
        if self.mode == GovernorMode.LIVE and invariants_violated:
            proposed = "stop_safe"
            reasons.append({"rule": "invariants_violation", "violations": invariants_violated})

        # Supervision withdrawal recommendation under sustained overload or sustained unknowns
        withdraw = False
        if pressure >= 0.9 and self.mode != GovernorMode.SHADOW:
            withdraw = True
            reasons.append(
                {
                    "rule": "withdraw_supervision_recommended",
                    "note": "gate pressure extremely high; recommend downgrading mode",
                }
            )

        # Gate creation conditions
        requires_gate = False
        if self.mode == GovernorMode.LIVE:
            if band in self.policy.human_gate_bands_live:
                requires_gate = True
            if knowledge_hole:
                requires_gate = True
            if withdraw:
                requires_gate = True

        # Create a gate if needed and none active, respecting cooldown
        if requires_gate and not self._active_gates():
            if self._gate_creation_allowed(world.tick, pressure, band, knowledge_hole):
                expires = 3 if (band == "STOP" or withdraw) else 5
                stage = 2 if withdraw else 1
                self._new_gate(
                    risk=risk,
                    world=world,
                    proposed=proposed,
                    stage=stage,
                    expires_in_ticks=expires,
                    reason="withdraw_supervision" if withdraw else ("knowledge_hole" if knowledge_hole else "risk_band"),
                )
                reasons.append({"rule": "human_gate_created", "expires_in_ticks": expires, "stage": stage})
            else:
                reasons.append(
                    {
                        "rule": "human_gate_suppressed",
                        "note": "cooldown active to prevent operator fatigue",
                    }
                )

        current_gate = self._select_current_gate()
        requires_gate = current_gate is not None
        human_gate_id = current_gate.gate_id if current_gate else None

        # Effective action considering mode and gate state
        if self.mode == GovernorMode.SHADOW:
            action = "none"
            reasons.append({"rule": "shadow_mode", "note": "no actuation; proposals only"})
        else:
            if withdraw:
                action = "withdraw_supervision"
            elif requires_gate:
                action = "hold_for_approval"
            else:
                action = proposed

        return Decision(
            tick=world.tick,
            action=action,
            proposed_action=proposed,
            band=band,
            requires_human_gate=requires_gate,
            human_gate_id=human_gate_id,
            invariants_violated=invariants_violated,
            invariant_proofs=invariant_proofs,
            reason_chain=reasons,
            gate_pressure_index=round(pressure, 3),
            knowledge_hole=knowledge_hole,
            epistemic=round(epistemic.epistemic, 3),
            safety_credit=round(float(risk.features.get("safety_credit", 0.0)), 3),
        )

    def apply_gate(
        self,
        gate_id: str,
        approved: bool,
        operator_id: Optional[str] = None,
        note: str = "",
    ) -> Dict[str, Any]:
        gate = self._gates.get(gate_id)
        if gate is None or gate.approved is not None:
            return {
                "override_applied": False,
                "error": "no_matching_gate",
                "gate_id": gate_id,
            }
        gate.approved = approved
        gate.operator_id = operator_id
        gate.note = note
        gate.resolved_at = utc_now_iso()
        self._last_gate_resolved_tick = gate.tick
        return {"override_applied": True, "gate": asdict(gate)}

    def get_pending_gates(self) -> List[HumanGateState]:
        return self._active_gates()

    def export_state(self) -> Dict[str, Any]:
        return {
            "asset_key": self.asset_key,
            "mode": self.mode,
            "_next_gate_id": self._next_gate_id,
            "_gates": {gid: asdict(g) for gid, g in self._gates.items()},
            "_gate_created_ticks": list(self._gate_created_ticks),
            "_last_gate_resolved_tick": self._last_gate_resolved_tick,
            "gate_pressure_cfg": asdict(self.gate_pressure_cfg),
        }

    def load_state(self, blob: Dict[str, Any]) -> None:
        self.asset_key = str(blob.get("asset_key", self.asset_key))
        self.mode = str(blob.get("mode", self.mode))
        self._next_gate_id = int(blob.get("_next_gate_id", self._next_gate_id))
        self._gate_created_ticks = [int(x) for x in blob.get("_gate_created_ticks", [])]
        self._last_gate_resolved_tick = blob.get("_last_gate_resolved_tick", self._last_gate_resolved_tick)
        # Gates
        self._gates = {}
        for gid, g in dict(blob.get("_gates", {})).items():
            self._gates[gid] = HumanGateState(**g)
        # Gate pressure cfg
        if "gate_pressure_cfg" in blob:
            self.gate_pressure_cfg = GatePressureConfig(**blob["gate_pressure_cfg"])


# ============================================================================
# Avalon – structured oversight items + scoring
# ============================================================================

@dataclass
class OversightContext:
    asset: AssetId
    scenario: str
    world: Dict[str, Any]
    risk: RiskPacket
    epistemic: EpistemicPacket
    decision: Decision
    policy_id: str
    shadow_diffs: List[Dict[str, Any]]


@dataclass
class OversightItem:
    summary: str
    risks: List[str]
    recommendations: List[str]
    confidence: float
    tags: List[str]


class AgentFn(Protocol):
    def __call__(self, ctx: OversightContext) -> OversightItem:
        ...


@dataclass
class Agent:
    name: str
    role: str  # "responder" | "scribe"
    fn: AgentFn
    enabled: bool = True

    def respond(self, ctx: OversightContext) -> OversightItem:
        return self.fn(ctx)


class Judge:
    def __init__(self, name: str = "DeterministicJudgeV4") -> None:
        self.name = name

    def score(self, item: OversightItem, context: Dict[str, Any]) -> Dict[str, float]:
        text = (item.summary or "") + " " + " ".join(item.recommendations or [])
        words = text.split()
        length = len(words)

        contains_risk_words = any(
            w in text.lower()
            for w in ["crash", "failure", "unsafe", "catastrophic", "ignore", "bypass", "override"]
        )
        contains_safety_words = any(
            w in text.lower()
            for w in ["monitor", "pause", "review", "human", "safety", "limit", "rollback", "halt", "hold"]
        )

        length_score = max(0.0, min(1.0, length / 240.0))
        structure_score = 1.0 if any(ch in text for ch in ["\n-", "\n1.", "\n*"]) else 0.6
        safety_bias = 0.85 if contains_safety_words else 0.45
        risk_penalty = 0.65 if contains_risk_words else 1.0

        clarity_raw = (length_score * 0.4 + structure_score * 0.3 + safety_bias * 0.3)
        clarity_raw *= risk_penalty
        clarity_raw = max(0.1, min(0.99, clarity_raw))

        disagreement = float(context.get("disagreement", 0.0))
        epistemic = float(context.get("epistemic", 0.0))
        # Epistemic makes text riskier unless it explicitly calls for HOLD/REVIEW
        epistemic_penalty = 0.0
        if epistemic >= 0.60 and ("hold" not in text.lower()) and ("review" not in text.lower()):
            epistemic_penalty = 10.0

        base_risk = (1.0 - clarity_raw) * 100.0
        risk_value = clamp(base_risk + disagreement * 0.5 + epistemic_penalty, 0.0, 100.0)
        overall = int(10 + clarity_raw * 89)

        return {
            "clarity": round(clarity_raw * 100, 1),
            "risk": round(risk_value, 1),
            "overall": float(overall),
            "length_score": round(length_score * 100, 1),
            "structure_score": round(structure_score * 100, 1),
        }


@dataclass
class ActionProposal:
    run_id: int
    description: str
    scope: str
    severity: str
    rollback_plan: str
    origin_agent: str


class AvalonEngine:
    def __init__(self, audit: AuditSpine, policy: SafetyPolicy) -> None:
        self.audit = audit
        self.policy = policy
        self.responders: List[Agent] = []
        self.scribes: List[Agent] = []
        self.judges: List[Judge] = [Judge()]
        self.run_id: int = 0

    def add_responder(self, agent: Agent) -> None:
        self.responders.append(agent)

    def add_scribe(self, agent: Agent) -> None:
        self.scribes.append(agent)

    def run(
        self,
        asset: AssetId,
        scenario: str,
        world: WorldState,
        risk: RiskPacket,
        epistemic: EpistemicPacket,
        decision: Decision,
        policy_id: str,
        shadow_diffs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        self.run_id += 1
        run_id = self.run_id

        ctx = OversightContext(
            asset=asset,
            scenario=scenario,
            world=world.to_dict(),
            risk=risk,
            epistemic=epistemic,
            decision=decision,
            policy_id=policy_id,
            shadow_diffs=shadow_diffs,
        )

        # House I – responders
        responder_items: Dict[str, OversightItem] = {}
        for agent in self.responders:
            if agent.enabled:
                responder_items[agent.name] = agent.respond(ctx)

        self.audit.log(
            "avalon_responders",
            {
                "run_id": run_id,
                "asset": asdict(asset),
                "scenario": scenario,
                "outputs": {k: asdict(v) for k, v in responder_items.items()},
            },
        )

        # House II – scribes (synthesis over responders)
        scribe_items: Dict[str, OversightItem] = {}
        for scribe in self.scribes:
            if scribe.enabled:
                scribe_items[scribe.name] = scribe.respond(ctx)

        self.audit.log(
            "avalon_scribes",
            {
                "run_id": run_id,
                "asset": asdict(asset),
                "outputs": {k: asdict(v) for k, v in scribe_items.items()},
            },
        )

        # House III – judges
        all_items: Dict[str, OversightItem] = {**responder_items, **scribe_items}
        lengths = [
            len(((item.summary or "") + " " + " ".join(item.recommendations or [])).split())
            for item in all_items.values()
        ] or [1]

        disagreement = float(pstdev(lengths)) if len(lengths) > 1 else 0.0
        disagreement = round(disagreement, 5)

        scores: Dict[str, Dict[str, float]] = {}
        for name, item in all_items.items():
            judge_scores = [j.score(item, {"disagreement": disagreement, "epistemic": epistemic.epistemic}) for j in self.judges]
            merged = {
                key: mean(s[key] for s in judge_scores)
                for key in ["clarity", "risk", "overall", "length_score", "structure_score"]
            }
            scores[name] = merged

        self.audit.log(
            "avalon_scores",
            {
                "run_id": run_id,
                "asset": asdict(asset),
                "scores": scores,
                "disagreement": disagreement,
                "epistemic": round(epistemic.epistemic, 3),
            },
        )

        # House IV – gatekeeper selection (no actuation)
        if scores:
            risk_cap = self.policy.avalon_risk_cap
            safe_candidates = [name for name in scores if scores[name]["risk"] <= risk_cap]
            if safe_candidates:
                winner_name = max(safe_candidates, key=lambda n: (scores[n]["overall"], n))
            else:
                winner_name = min(scores.keys(), key=lambda n: (scores[n]["risk"], n))
            winning_item = all_items[winner_name]
            winning_score = scores[winner_name]
        else:
            winner_name = ""
            winning_item = OversightItem(summary="", risks=[], recommendations=[], confidence=0.5, tags=[])
            winning_score = {"clarity": 0.0, "risk": 100.0, "overall": 10.0, "length_score": 0.0, "structure_score": 0.0}

        clarity_now = float(winning_score["clarity"])
        disagreement_factor = min(100.0, disagreement)
        predicted_risk = min(100.0, float(winning_score["risk"]) + 0.3 * disagreement_factor + (90.0 - clarity_now) * 0.2)

        proposal = self._build_proposal(run_id, winner_name, winning_item, risk, decision)

        decision_packet = {
            "run_id": run_id,
            "asset": asdict(asset),
            "winner": winner_name,
            "item": asdict(winning_item),
            "scores": winning_score,
            "disagreement": round(disagreement, 3),
            "epistemic": round(epistemic.epistemic, 3),
            "predicted_risk": round(predicted_risk, 1),
            "proposal": asdict(proposal) if proposal else None,
        }
        self.audit.log("avalon_decision", decision_packet)

        return {
            "responders": {k: asdict(v) for k, v in responder_items.items()},
            "scribes": {k: asdict(v) for k, v in scribe_items.items()},
            "scores": scores,
            "decision": decision_packet,
        }

    def _build_proposal(
        self,
        run_id: int,
        agent_name: str,
        item: OversightItem,
        risk: RiskPacket,
        decision: Decision,
    ) -> Optional[ActionProposal]:
        if not agent_name or not (item.summary or "").strip():
            return None

        severity = "low"
        if risk.band in ("HOLD", "STOP") or decision.requires_human_gate or decision.knowledge_hole:
            severity = "high"
        elif risk.band == "WATCH":
            severity = "medium"

        description = (
            f"Run safety-supervised adjustment plan from {agent_name} in "
            f"{decision.action} mode (no direct actuation)."
        )
        rollback_plan = (
            "No autonomous actuation permitted. All outputs are proposals only; "
            "operators retain full control and can rollback by ignoring or "
            "superseding OmegaV4 recommendations."
        )
        return ActionProposal(
            run_id=run_id,
            description=description,
            scope="simulation_only",
            severity=severity,
            rollback_plan=rollback_plan,
            origin_agent=agent_name,
        )


# Demo structured responder / scribe implementations
def responder_structured(ctx: OversightContext) -> OversightItem:
    band = ctx.risk.band
    mode = ctx.world["mode"]
    top = ctx.risk.attribution.top_factors[:3]
    top_str = ", ".join([f"{t['factor']}:{t['delta']:+.3f}" for t in top]) if top else "n/a"

    summary = (
        f"Asset {ctx.asset.site}/{ctx.asset.asset} in {mode.upper()} mode at tick {ctx.risk.tick}: "
        f"band={band}, risk={ctx.risk.risk:.3f}, epistemic={ctx.epistemic.epistemic:.3f}. "
        f"Top Δ-risk drivers: {top_str}."
    )

    risks = [
        f"band={band}",
        f"epistemic={ctx.epistemic.epistemic:.2f}",
        f"gate_pressure={ctx.decision.gate_pressure_index:.2f}",
    ]
    if ctx.epistemic.knowledge_hole:
        risks.append("knowledge_hole=true")
    recommendations = [
        "If epistemic is high, prioritize data quality checks before policy tuning.",
        "Use Δ-risk drivers to explain why the band escalated; avoid arguing about the scalar.",
    ]
    tags = ["structured", "attribution", f"band:{band}", f"mode:{mode}"]
    return OversightItem(summary=summary, risks=risks, recommendations=recommendations, confidence=0.9, tags=tags)


def responder_conservative(ctx: OversightContext) -> OversightItem:
    band = ctx.risk.band
    risks = ["Prefer halt/hold over optimization", "Escalate early under uncertainty"]
    if ctx.epistemic.knowledge_hole:
        risks.append("Unknown state detected (knowledge hole)")
    recommendations = [
        "Pause any autonomy-side changes until operators review sensor health + spreads.",
        "Treat STOP band or knowledge holes as requiring explicit supervisor approval.",
        "If gate pressure is high, consider downgrading mode to TRAINING/SHADOW.",
    ]
    tags = ["conservative", "safety-first", f"band:{band}"]
    return OversightItem(
        summary=f"Conservative posture for {ctx.asset.site}/{ctx.asset.asset}: band={band}, human safety prioritized.",
        risks=risks,
        recommendations=recommendations,
        confidence=0.86,
        tags=tags,
    )


def responder_aggressive(ctx: OversightContext) -> OversightItem:
    band = ctx.risk.band
    risks = ["Over-optimization could increase operator load", "Model drift if policies change fast"]
    recommendations = [
        "Use OmegaV4 as an offline experiment engine with fault injection enabled.",
        "Compare shadow policies; select candidates that reduce gates without increasing STOP events.",
        "Only pursue throughput if epistemic uncertainty stays low and invariants have healthy margins.",
    ]
    tags = ["optimization", "analysis", f"band:{band}"]
    return OversightItem(
        summary=f"Optimization-oriented view for {ctx.asset.site}/{ctx.asset.asset}: band={band}, explore throughput under a hard safety floor.",
        risks=risks,
        recommendations=recommendations,
        confidence=0.8,
        tags=tags,
    )


def scribe_safety(ctx: OversightContext) -> OversightItem:
    band = ctx.risk.band
    risks = [
        "Human gates are required under high bands or knowledge holes in LIVE mode.",
        "Sensor health, contradictions, and multi-asset proximity can shrink safety margins.",
    ]
    recommendations = [
        "Audit human-gate decisions regularly (fatigue, override patterns, stage escalation).",
        "Prefer fixing telemetry quality over tuning thresholds when epistemic is elevated.",
    ]
    tags = ["scribe", "safety", f"band:{band}"]
    return OversightItem(
        summary=f"Safety synthesis for {ctx.asset.site}/{ctx.asset.asset}: band={band}, epistemic-aware oversight active.",
        risks=risks,
        recommendations=recommendations,
        confidence=0.9,
        tags=tags,
    )


def scribe_ops(ctx: OversightContext) -> OversightItem:
    band = ctx.risk.band
    diffs = ctx.shadow_diffs or []
    diff_note = ""
    if diffs:
        most_div = max(diffs, key=lambda d: abs(d.get("band_delta_rank", 0)))
        diff_note = f" Shadow policy divergence: {most_div.get('policy_id')} band_delta={most_div.get('band_delta')}."
    risks = [
        "Operator overload if gate frequency is high",
        "Policy drift across deployments",
        "Hidden unknowns if epistemic is ignored",
    ]
    recommendations = [
        "Deploy in SHADOW mode first; require replay capsules for incidents.",
        "Use gate pressure index as an ops KPI; treat sustained overload as a safety event.",
    ]
    tags = ["scribe", "ops", f"band:{band}"]
    return OversightItem(
        summary=f"Operational synthesis for {ctx.asset.site}/{ctx.asset.asset}: band={band}.{diff_note}",
        risks=risks,
        recommendations=recommendations,
        confidence=0.88,
        tags=tags,
    )


def register_demo_agents(avalon: AvalonEngine) -> None:
    avalon.add_responder(Agent("Responder: Structured", "responder", responder_structured))
    avalon.add_responder(Agent("Responder: Conservative", "responder", responder_conservative))
    avalon.add_responder(Agent("Responder: Aggressive", "responder", responder_aggressive))
    avalon.add_scribe(Agent("Scribe: Safety", "scribe", scribe_safety))
    avalon.add_scribe(Agent("Scribe: Operations", "scribe", scribe_ops))


# ============================================================================
# Memory – recap frames with hash chaining (V4 expanded)
# ============================================================================

@dataclass
class MemoryFrame:
    id: int
    asset_key: str
    timestamp: str
    tick: int
    mode: str
    summary: str
    key_topics: List[str]
    risk_band: str
    risk: float
    epistemic: float
    knowledge_hole: bool
    gate_pressure_index: float
    action: str
    predicted_risk: float
    winner_agent: str
    human_gate_pending: bool
    human_gate_id: Optional[str]
    tick_ms: float
    safety_credit: float
    top_risk_factors: List[Dict[str, Any]]
    policy_id: str
    policy_hash: str
    shadow_diffs: List[Dict[str, Any]]
    world_snapshot: Dict[str, Any]
    scenario: str
    hash: str
    prev_hash: str


class MemoryEngine:
    def __init__(self) -> None:
        self.frames: List[MemoryFrame] = []
        self._last_hash: str = "0" * 64
        self._next_id: int = 1

    def add_frame(
        self,
        asset_key: str,
        tick: int,
        mode: str,
        risk_band: str,
        risk_value: float,
        epistemic: float,
        knowledge_hole: bool,
        gate_pressure_index: float,
        action: str,
        predicted_risk: float,
        winner_agent: str,
        human_gate_pending: bool,
        human_gate_id: Optional[str],
        tick_ms: float,
        safety_credit: float,
        top_risk_factors: List[Dict[str, Any]],
        policy_id: str,
        policy_hash: str,
        shadow_diffs: List[Dict[str, Any]],
        world_snapshot: Dict[str, Any],
        scenario: str,
    ) -> MemoryFrame:
        ts = utc_now_iso()
        summary = (
            f"Asset {asset_key} tick {tick} in {mode} mode; band={risk_band}; "
            f"risk={risk_value:.3f}; epistemic={epistemic:.3f}; action={action}; "
            f"pred_risk={predicted_risk:.1f}; gate_pending={human_gate_pending}; gate_id={human_gate_id or 'none'}"
        )
        key_topics = [
            f"asset:{asset_key}",
            f"band:{risk_band}",
            f"mode:{mode}",
            "knowledge_hole" if knowledge_hole else "no_knowledge_hole",
            "gate_pending" if human_gate_pending else "no_gate",
        ]

        frame_dict = {
            "id": self._next_id,
            "asset_key": asset_key,
            "timestamp": ts,
            "tick": tick,
            "mode": mode,
            "summary": summary,
            "key_topics": key_topics,
            "risk_band": risk_band,
            "risk": risk_value,
            "epistemic": epistemic,
            "knowledge_hole": knowledge_hole,
            "gate_pressure_index": gate_pressure_index,
            "action": action,
            "predicted_risk": predicted_risk,
            "winner_agent": winner_agent,
            "human_gate_pending": human_gate_pending,
            "human_gate_id": human_gate_id,
            "tick_ms": tick_ms,
            "safety_credit": safety_credit,
            "top_risk_factors": top_risk_factors,
            "policy_id": policy_id,
            "policy_hash": policy_hash,
            "shadow_diffs": shadow_diffs,
            "world_snapshot": world_snapshot,
            "scenario": scenario,
            "prev_hash": self._last_hash,
        }
        h = sha256_json(frame_dict)
        frame = MemoryFrame(
            id=self._next_id,
            asset_key=asset_key,
            timestamp=ts,
            tick=tick,
            mode=mode,
            summary=summary,
            key_topics=key_topics,
            risk_band=risk_band,
            risk=risk_value,
            epistemic=epistemic,
            knowledge_hole=knowledge_hole,
            gate_pressure_index=gate_pressure_index,
            action=action,
            predicted_risk=predicted_risk,
            winner_agent=winner_agent,
            human_gate_pending=human_gate_pending,
            human_gate_id=human_gate_id,
            tick_ms=tick_ms,
            safety_credit=safety_credit,
            top_risk_factors=top_risk_factors,
            policy_id=policy_id,
            policy_hash=policy_hash,
            shadow_diffs=shadow_diffs,
            world_snapshot=world_snapshot,
            scenario=scenario,
            prev_hash=self._last_hash,
            hash=h,
        )
        self.frames.append(frame)
        self._last_hash = h
        self._next_id += 1
        return frame

    def tail(self, n: int = 16, asset_key: Optional[str] = None) -> List[MemoryFrame]:
        if asset_key is None:
            return self.frames[-n:]
        filtered = [f for f in self.frames if f.asset_key == asset_key]
        return filtered[-n:]

    def verify_chain(self) -> Tuple[bool, Optional[int]]:
        prev = "0" * 64
        for idx, f in enumerate(self.frames):
            frame_dict = {
                "id": f.id,
                "asset_key": f.asset_key,
                "timestamp": f.timestamp,
                "tick": f.tick,
                "mode": f.mode,
                "summary": f.summary,
                "key_topics": f.key_topics,
                "risk_band": f.risk_band,
                "risk": f.risk,
                "epistemic": f.epistemic,
                "knowledge_hole": f.knowledge_hole,
                "gate_pressure_index": f.gate_pressure_index,
                "action": f.action,
                "predicted_risk": f.predicted_risk,
                "winner_agent": f.winner_agent,
                "human_gate_pending": f.human_gate_pending,
                "human_gate_id": f.human_gate_id,
                "tick_ms": f.tick_ms,
                "safety_credit": f.safety_credit,
                "top_risk_factors": f.top_risk_factors,
                "policy_id": f.policy_id,
                "policy_hash": f.policy_hash,
                "shadow_diffs": f.shadow_diffs,
                "world_snapshot": f.world_snapshot,
                "scenario": f.scenario,
                "prev_hash": prev,
            }
            expected_hash = sha256_json(frame_dict)
            if f.prev_hash != prev or f.hash != expected_hash:
                return False, idx
            prev = f.hash
        return True, None

    def to_json(self) -> str:
        serializable = [asdict(f) for f in self.frames]
        return json.dumps(serializable, indent=2, sort_keys=True, ensure_ascii=False)

    def load_from_json(self, raw: str) -> None:
        data = json.loads(raw)
        frames: List[MemoryFrame] = []
        for f in data:
            frames.append(MemoryFrame(**f))
        self.frames = frames
        if frames:
            self._next_id = int(frames[-1].id) + 1
            self._last_hash = frames[-1].hash
        else:
            self._next_id = 1
            self._last_hash = "0" * 64


# ============================================================================
# Policy shadow diff (V4)
# ============================================================================

@dataclass
class PolicyShadowResult:
    policy_id: str
    risk: float
    band: str
    would_gate_live: bool


def band_rank(b: str) -> int:
    return {"LOW": 0, "WATCH": 1, "HOLD": 2, "STOP": 3}.get(b, 0)


def compute_policy_shadow(
    shadow_policies: Dict[str, SafetyPolicy],
    shadow_kernels: Dict[str, SafetyKernel],
    shadow_states: Dict[str, RiskState],
    world: WorldState,
    ctx: RiskContext,
    mode: str,
) -> List[PolicyShadowResult]:
    results: List[PolicyShadowResult] = []
    for pid, pol in shadow_policies.items():
        kernel = shadow_kernels[pid]
        state = shadow_states.setdefault(pid, RiskState())
        pkt, _, _, _ = kernel.eval(world, ctx, state)
        would_gate = (mode == GovernorMode.LIVE) and (pkt.band in pol.human_gate_bands_live)
        results.append(PolicyShadowResult(policy_id=pid, risk=float(pkt.risk), band=pkt.band, would_gate_live=would_gate))
    return results


def shadow_diffs_from_results(active_band: str, active_would_gate: bool, shadow: List[PolicyShadowResult]) -> List[Dict[str, Any]]:
    diffs: List[Dict[str, Any]] = []
    active_rank = band_rank(active_band)
    for r in shadow:
        diffs.append(
            {
                "policy_id": r.policy_id,
                "band": r.band,
                "risk": round(r.risk, 3),
                "would_gate_live": r.would_gate_live,
                "band_delta": band_rank(r.band) - active_rank,
                "band_delta_rank": band_rank(r.band) - active_rank,
                "gate_delta": int(r.would_gate_live) - int(active_would_gate),
            }
        )
    return diffs


# ============================================================================
# Replay capsule (V4)
# ============================================================================

def world_from_dict(d: Dict[str, Any]) -> WorldState:
    """Reconstruct a WorldState (and nested dataclasses) from a dict produced by asdict()."""
    asset = AssetId(**d["asset"])
    vehicle = VehicleState(**d.get("vehicle", {}))
    pose = Pose(**d.get("pose", {}))
    sensors_dict = d.get("sensors", {})
    drift = [SensorChannel(**c) for c in sensors_dict.get("drift", [])]
    speed = [SensorChannel(**c) for c in sensors_dict.get("speed", [])]
    stability = [SensorChannel(**c) for c in sensors_dict.get("stability", [])]
    sensors = SensorSuite(drift=drift, speed=speed, stability=stability)
    return WorldState(
        asset=asset,
        tick=int(d.get("tick", 0)),
        mode=str(d.get("mode", GovernorMode.SHADOW)),
        vehicle=vehicle,
        pose=pose,
        sensors=sensors,
    )


def build_replay_capsule(omega: "OmegaV4", max_audit_entries: Optional[int] = None, max_memory_frames: Optional[int] = None) -> str:
    # IMPORTANT: to preserve hash-chain verifiability, export from the start of each chain.
    audit_entries = list(omega.audit.entries)
    memory_frames = list(omega.memory.frames)
    if isinstance(max_audit_entries, int) and max_audit_entries > 0 and len(audit_entries) > max_audit_entries:
        audit_entries = audit_entries[:max_audit_entries]
    if isinstance(max_memory_frames, int) and max_memory_frames > 0 and len(memory_frames) > max_memory_frames:
        memory_frames = memory_frames[:max_memory_frames]

    capsule = {
        "version": "OmegaV4",
        "created_at": utc_now_iso(),
        "session_id": omega.audit.session_id,
        "mode": omega.mode,
        "policy_active": json.loads(DEFAULT_POLICY_JSON_V4),
        "policy_active_source_hash": omega.policy.source_hash,
        "shadow_policies": {pid: {"id": p.id, "risk_cfg": asdict(p.risk_cfg), "invariants": asdict(p.invariants),
                                 "band_to_action": p.band_to_action, "human_gate_bands_live": list(p.human_gate_bands_live),
                                 "avalon_risk_cap": p.avalon_risk_cap, "source_hash": p.source_hash}
                           for pid, p in omega.shadow_policies.items()},
        "fault_profile": asdict(omega.fault_profile),
        "assets": {
            key: {
                "asset": asdict(ctx.asset),
                "seed": ctx.seed,
                "world": ctx.world.to_dict(),
                "risk_state": asdict(ctx.risk_state),
                "shadow_risk_states": {pid: asdict(rs) for pid, rs in ctx.shadow_risk_states.items()},
                "governor_state": ctx.governor.export_state(),
            }
            for key, ctx in omega.assets.items()
        },
        "audit_entries_json": json.dumps(
            [
                {
                    "session_id": e.session_id,
                    "seq": e.seq,
                    "timestamp": e.timestamp,
                    "kind": e.kind,
                    "payload": e.payload,
                    "prev_hash": e.prev_hash,
                    "hash": e.hash,
                }
                for e in audit_entries
            ],
            ensure_ascii=False,
        ),
        "memory_frames_json": json.dumps([asdict(f) for f in memory_frames], ensure_ascii=False),
    }
    return json.dumps(capsule, indent=2, sort_keys=True, ensure_ascii=False)


def load_replay_capsule(raw: str) -> "OmegaV4":
    capsule = json.loads(raw)
    if capsule.get("version") != "OmegaV4":
        raise ValueError("Replay capsule version mismatch")

    omega = OmegaV4(mode=str(capsule.get("mode", GovernorMode.SHADOW)), session_id=str(capsule.get("session_id", "")) or None)

    # Restore shadow policies (if present in capsule)
    if "shadow_policies" in capsule and isinstance(capsule.get("shadow_policies"), dict):
        sp: Dict[str, SafetyPolicy] = {}
        for pid_key, p in dict(capsule.get("shadow_policies", {})).items():
            try:
                cfg = {
                    "id": str(p.get("id", pid_key)),
                    "risk_cfg": dict(p.get("risk_cfg", {})),
                    "invariants": dict(p.get("invariants", {})),
                    "band_to_action": dict(p.get("band_to_action", {})),
                    "human_gate_bands_live": list(p.get("human_gate_bands_live", ["HOLD", "STOP"])),
                    "avalon_risk_cap": float(p.get("avalon_risk_cap", 65.0)),
                }
                pol = _build_policy_from_dict(cfg)
                sp[pol.id] = pol
            except Exception:
                continue
        if sp:
            omega.shadow_policies = sp
            omega.shadow_kernels = {pid: SafetyKernel(p.risk_cfg) for pid, p in omega.shadow_policies.items()}

    # Restore fault profile
    omega.fault_profile = FaultProfile(**capsule.get("fault_profile", {}))

    # Restore audit + memory
    omega.audit.load_from_json(str(capsule.get("audit_entries_json", "[]")))
    omega.memory.load_from_json(str(capsule.get("memory_frames_json", "[]")))

    # Restore assets
    omega.assets = {}
    for key, blob in dict(capsule.get("assets", {})).items():
        asset = AssetId(**blob["asset"])
        world = world_from_dict(blob["world"])
        seed = int(blob["seed"])
        governor = Governor(omega.policy, mode=omega.mode, asset_key=key)
        governor.load_state(blob.get("governor_state", {}))
        ctx = AssetContext(asset=asset, world=world, seed=seed, governor=governor)
        ctx.risk_state = RiskState(**blob.get("risk_state", {}))
        # shadow states
        ctx.shadow_risk_states = {pid: RiskState(**rs) for pid, rs in dict(blob.get("shadow_risk_states", {})).items()}
        omega.assets[key] = ctx

    omega.audit.log("replay_capsule_loaded", {"assets": list(omega.assets.keys()), "mode": omega.mode})
    return omega


# ============================================================================
# OmegaV4 – orchestrator (multi-asset)
# ============================================================================

@dataclass
class TickResult:
    asset: AssetId
    world: WorldState
    risk: RiskPacket
    epistemic: EpistemicPacket
    decision: Decision
    avalon: Dict[str, Any]
    tick_ms: float
    scenario: str
    shadow_diffs: List[Dict[str, Any]]


@dataclass
class AssetContext:
    asset: AssetId
    world: WorldState
    seed: int
    governor: Governor
    risk_state: RiskState = field(default_factory=RiskState)
    shadow_risk_states: Dict[str, RiskState] = field(default_factory=dict)
    last_tick_ms: float = 0.0
    last_risk_band: str = "LOW"
    last_predicted_risk: float = 0.0
    last_epistemic: float = 0.0
    last_gate_pressure: float = 0.0
    last_knowledge_hole: bool = False


class OmegaV4:
    def __init__(self, mode: str = GovernorMode.SHADOW, session_id: Optional[str] = None) -> None:
        self.audit = AuditSpine(session_id=session_id)
        self.policy = load_default_policy()
        self.kernel = SafetyKernel(self.policy.risk_cfg)
        self.epistemic_cfg = EpistemicConfig()
        self.avalon = AvalonEngine(self.audit, self.policy)
        register_demo_agents(self.avalon)
        self.memory = MemoryEngine()
        self.mode = mode

        # Multi-asset registry
        self.assets: Dict[str, AssetContext] = {}
        self.fault_profile = FaultProfile()
        self.last_interactions: Dict[str, InteractionSummary] = {}

        # Policy shadowing
        self.shadow_policies: Dict[str, SafetyPolicy] = build_shadow_policies(self.policy)
        self.shadow_kernels: Dict[str, SafetyKernel] = {pid: SafetyKernel(p.risk_cfg) for pid, p in self.shadow_policies.items()}

        # Session init event
        self.audit.log(
            "session_init",
            {
                "session_id": self.audit.session_id,
                "policy_id": self.policy.id,
                "policy_source_hash": self.policy.source_hash,
                "mode": mode,
                "shadow_policies": list(self.shadow_policies.keys()),
            },
        )

    def ensure_asset(self, site: str, asset_name: str) -> AssetId:
        site = site or "site-1"
        asset_name = asset_name or "asset-1"
        asset = AssetId(site=site, asset=asset_name)
        key = asset.key()

        if key not in self.assets:
            seed = random.randint(1, 2**31 - 1)
            world = WorldState(asset=asset, tick=0, mode=self.mode)
            governor = Governor(self.policy, mode=self.mode, asset_key=key)
            ctx = AssetContext(asset=asset, world=world, seed=seed, governor=governor)

            # Initialize shadow policy states
            for pid in self.shadow_policies.keys():
                ctx.shadow_risk_states[pid] = RiskState()

            self.assets[key] = ctx
            self.audit.log("asset_registered", {"asset": asdict(asset), "seed": seed})

        return asset

    def list_assets(self) -> List[str]:
        return sorted(self.assets.keys())

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        for ctx in self.assets.values():
            ctx.world.mode = mode
            ctx.governor.mode = mode
        self.audit.log("mode_changed", {"mode": mode})

    def snapshot_asset(self, key: str) -> Optional[Dict[str, Any]]:
        ctx = self.assets.get(key)
        if ctx is None:
            return None
        return {
            "asset": asdict(ctx.asset),
            "seed": ctx.seed,
            "tick": ctx.world.tick,
            "mode": ctx.world.mode,
            "world": ctx.world.to_dict(),
            "tick_ms": ctx.last_tick_ms,
            "risk_band": ctx.last_risk_band,
            "predicted_risk": ctx.last_predicted_risk,
            "epistemic": ctx.last_epistemic,
            "gate_pressure_index": ctx.last_gate_pressure,
            "knowledge_hole": ctx.last_knowledge_hole,
            "safety_credit": ctx.risk_state.safety_credit,
        }

    def _update_safety_credit(self, state: RiskState, risk: RiskPacket, epistemic: EpistemicPacket) -> None:
        """
        Safety credit/debt ledger update.

        - Low risk accrues credit
        - HOLD/STOP burns credit
        - Knowledge holes burn credit aggressively
        """
        delta = 0.0
        if epistemic.knowledge_hole:
            delta -= 1.5
            state.knowledge_hole_streak += 1
        else:
            state.knowledge_hole_streak = 0

        if risk.band == "LOW":
            delta += 0.25 if risk.risk < 0.20 else 0.10
        elif risk.band == "WATCH":
            delta += 0.05
        elif risk.band == "HOLD":
            delta -= 0.75
        elif risk.band == "STOP":
            delta -= 1.25

        # Small burn for repeated volatility
        if state.history_risk_count > 3:
            avg = state.history_risk_sum / state.history_risk_count
            if avg > 0.55 and risk.risk > 0.55:
                delta -= 0.10

        state.safety_credit = clamp(state.safety_credit + delta, -50.0, 50.0)

    def tick_all(self, scenario: str) -> Dict[str, TickResult]:
        results: Dict[str, TickResult] = {}
        if not self.assets:
            return results

        # Step 1: increment ticks and simulate dynamics for all assets
        for ctx in self.assets.values():
            ctx.world.tick += 1
            ctx.world = simulate_dynamics(ctx.world, ctx.seed, self.fault_profile)

        # Step 2: compute multi-asset interactions
        worlds = {key: ctx.world for key, ctx in self.assets.items()}
        interactions = compute_interactions(worlds)
        self.last_interactions = interactions

        # Step 3: per-asset sensors + risk + epistemic + governor + avalon + memory + audit
        for key, ctx in self.assets.items():
            start_ns = time.perf_counter_ns()

            update_sensors(ctx.world, ctx.seed, self.fault_profile)
            metrics = compute_sensor_metrics(ctx.world)

            inter = interactions[key]
            proximity_norm = clamp(1.0 - inter.nearest_distance / MAX_PROX_DISTANCE, 0.0, 1.0)

            risk_ctx = RiskContext(
                sensor_health=metrics.sensor_health,
                stale_fraction=metrics.stale_fraction,
                contradictory_fraction=metrics.contradictory_fraction,
                proximity=proximity_norm,
                same_zone_conflict=inter.same_zone_conflict,
            )

            # Safety kernel
            risk, drift_spread, speed_spread, stability_spread = self.kernel.eval(ctx.world, risk_ctx, ctx.risk_state)

            # Epistemic packet
            epistemic = compute_epistemic(
                tick=ctx.world.tick,
                metrics=metrics,
                drift_spread=drift_spread,
                speed_spread=speed_spread,
                stability_spread=stability_spread,
                cfg=self.epistemic_cfg,
            )

            # Update safety credit after risk + epistemic computed (affects next tick)
            self._update_safety_credit(ctx.risk_state, risk, epistemic)

            # Policy shadow diff
            shadow_results = compute_policy_shadow(
                shadow_policies=self.shadow_policies,
                shadow_kernels=self.shadow_kernels,
                shadow_states=ctx.shadow_risk_states,
                world=ctx.world,
                ctx=risk_ctx,
                mode=self.mode,
            )
            active_would_gate = (self.mode == GovernorMode.LIVE) and (risk.band in self.policy.human_gate_bands_live)
            shadow_diffs = shadow_diffs_from_results(risk.band, active_would_gate, shadow_results)

            # Governor (provisional tick_ms placeholder)
            decision = ctx.governor.evaluate(risk=risk, world=ctx.world, epistemic=epistemic, tick_ms=0.0)

            # Avalon
            avalon_result = self.avalon.run(
                asset=ctx.asset,
                scenario=scenario,
                world=ctx.world,
                risk=risk,
                epistemic=epistemic,
                decision=decision,
                policy_id=self.policy.id,
                shadow_diffs=shadow_diffs,
            )

            end_ns = time.perf_counter_ns()
            tick_ms = (end_ns - start_ns) / 1_000_000.0
            ctx.last_tick_ms = tick_ms

            # Re-evaluate governor with actual tick duration for invariant proofs
            decision = ctx.governor.evaluate(risk=risk, world=ctx.world, epistemic=epistemic, tick_ms=tick_ms)

            dec_packet = avalon_result["decision"]
            predicted_risk = float(dec_packet["predicted_risk"])
            winner_agent = str(dec_packet["winner"])

            ctx.last_risk_band = risk.band
            ctx.last_predicted_risk = predicted_risk
            ctx.last_epistemic = float(epistemic.epistemic)
            ctx.last_gate_pressure = float(decision.gate_pressure_index)
            ctx.last_knowledge_hole = bool(epistemic.knowledge_hole)

            # Memory frame
            frame = self.memory.add_frame(
                asset_key=key,
                tick=ctx.world.tick,
                mode=ctx.world.mode,
                risk_band=risk.band,
                risk_value=float(risk.risk),
                epistemic=float(epistemic.epistemic),
                knowledge_hole=bool(epistemic.knowledge_hole),
                gate_pressure_index=float(decision.gate_pressure_index),
                action=decision.action,
                predicted_risk=predicted_risk,
                winner_agent=winner_agent,
                human_gate_pending=decision.requires_human_gate,
                human_gate_id=decision.human_gate_id,
                tick_ms=tick_ms,
                safety_credit=float(ctx.risk_state.safety_credit),
                top_risk_factors=risk.attribution.top_factors,
                policy_id=self.policy.id,
                policy_hash=self.policy.source_hash,
                shadow_diffs=shadow_diffs,
                world_snapshot=ctx.world.to_dict(),
                scenario=scenario,
            )

            # Audit
            self.audit.log(
                "omega_tick",
                {
                    "asset": asdict(ctx.asset),
                    "asset_key": key,
                    "tick": ctx.world.tick,
                    "duration_ms": tick_ms,
                    "world": ctx.world.to_dict(),
                    "risk": asdict(risk),
                    "epistemic": asdict(epistemic),
                    "decision": asdict(decision),
                    "predicted_risk": predicted_risk,
                    "winner_agent": winner_agent,
                    "scenario": scenario,
                    "policy_id": self.policy.id,
                    "policy_hash": self.policy.source_hash,
                    "shadow_diffs": shadow_diffs,
                    "memory_frame_id": frame.id,
                },
            )

            results[key] = TickResult(
                asset=ctx.asset,
                world=ctx.world,
                risk=risk,
                epistemic=epistemic,
                decision=decision,
                avalon=avalon_result,
                tick_ms=tick_ms,
                scenario=scenario,
                shadow_diffs=shadow_diffs,
            )

        return results

    def verify_integrity(self) -> Dict[str, Any]:
        audit_ok, audit_idx = self.audit.verify_chain()
        mem_ok, mem_idx = self.memory.verify_chain()
        return {
            "audit_ok": audit_ok,
            "audit_first_bad_index": audit_idx,
            "memory_ok": mem_ok,
            "memory_first_bad_index": mem_idx,
        }

    def get_gate_queue(self) -> List[HumanGateState]:
        queue: List[HumanGateState] = []
        for ctx in self.assets.values():
            queue.extend(ctx.governor.get_pending_gates())
        return queue

    def export_replay_capsule(self) -> str:
        return build_replay_capsule(self)


# ============================================================================
# Streamlit UI – OmegaV4 oversight console
# ============================================================================

st.set_page_config(page_title="OmegaV4 – Unified Autonomous Oversight Kernel", layout="wide")


def init_session() -> None:
    if "omega" not in st.session_state:
        st.session_state.omega = OmegaV4(mode=GovernorMode.SHADOW)
        st.session_state.clarity_hist: Dict[str, List[float]] = {}
        st.session_state.risk_hist: Dict[str, List[float]] = {}
        st.session_state.pred_risk_hist: Dict[str, List[float]] = {}
        st.session_state.epistemic_hist: Dict[str, List[float]] = {}
        st.session_state.gate_pressure_hist: Dict[str, List[float]] = {}

        # Bootstrap a default asset
        omega: OmegaV4 = st.session_state.omega
        omega.ensure_asset("site-1", "vehicle-1")


init_session()
omega: OmegaV4 = st.session_state.omega  # type: ignore[assignment]

# --- sidebar configuration -------------------------------------------------
st.sidebar.header("OmegaV4 Configuration")

mode = st.sidebar.selectbox(
    "Governor mode (global)",
    [GovernorMode.SHADOW, GovernorMode.TRAINING, GovernorMode.LIVE],
    index=[GovernorMode.SHADOW, GovernorMode.TRAINING, GovernorMode.LIVE].index(omega.mode),
)
if mode != omega.mode:
    omega.set_mode(mode)

st.sidebar.markdown("#### Assets")
site_input = st.sidebar.text_input("Site ID", value="site-1")
asset_input = st.sidebar.text_input("Asset ID", value="vehicle-1")
if st.sidebar.button("Ensure asset exists"):
    omega.ensure_asset(site_input.strip(), asset_input.strip())

asset_keys = omega.list_assets()
if not asset_keys:
    st.sidebar.warning("No assets registered yet; creating default.")
    omega.ensure_asset("site-1", "vehicle-1")
    asset_keys = omega.list_assets()

active_asset_key = st.sidebar.selectbox("Active asset", options=asset_keys, index=0)

policy_exp = st.sidebar.expander("Active safety policy", expanded=False)
with policy_exp:
    st.code(
        json.dumps(
            {
                "id": omega.policy.id,
                "source_hash": omega.policy.source_hash,
                "risk_cfg": asdict(omega.policy.risk_cfg),
                "invariants": asdict(omega.policy.invariants),
                "band_to_action": omega.policy.band_to_action,
                "human_gate_bands_live": omega.policy.human_gate_bands_live,
                "avalon_risk_cap": omega.policy.avalon_risk_cap,
                "shadow_policies": list(omega.shadow_policies.keys()),
            },
            indent=2,
            sort_keys=True,
        ),
        language="json",
    )

fault_exp = st.sidebar.expander("Fault injection (simulation only)", expanded=False)
with fault_exp:
    fp = omega.fault_profile
    fp.drift_spike_prob = st.slider("Drift spike probability", 0.0, 0.5, float(fp.drift_spike_prob), 0.01)
    fp.dropout_prob = st.slider("Telemetry dropout probability", 0.0, 0.5, float(fp.dropout_prob), 0.01)
    fp.freeze_prob = st.slider("Sensor freeze probability", 0.0, 0.5, float(fp.freeze_prob), 0.01)
    fp.negative_speed_prob = st.slider("Negative speed probability", 0.0, 0.5, float(fp.negative_speed_prob), 0.01)
    fp.contradictory_prob = st.slider("Contradictory reading probability", 0.0, 0.5, float(fp.contradictory_prob), 0.01)
    fp.timestamp_anomaly_prob = st.slider("Timestamp anomaly probability", 0.0, 0.5, float(fp.timestamp_anomaly_prob), 0.01)

st.sidebar.markdown("#### Thresholds")
risk_threshold = st.sidebar.slider("Predicted text-risk threshold (alert)", 0, 100, 60, 5)
clarity_target = st.sidebar.slider("Target clarity (%)", 0, 100, 85, 5)
epistemic_threshold = st.sidebar.slider("Epistemic threshold (knowledge hole)", 0.0, 1.0, float(omega.epistemic_cfg.knowledge_hole_threshold), 0.01)
omega.epistemic_cfg.knowledge_hole_threshold = epistemic_threshold

st.sidebar.markdown("---")

# Replay capsule import/export
replay_exp = st.sidebar.expander("Replay capsule (export/import)", expanded=False)
with replay_exp:
    if st.button("Prepare replay capsule JSON"):
        capsule = omega.export_replay_capsule()
        st.session_state._replay_capsule_json = capsule  # stash

    capsule_data = st.session_state.get("_replay_capsule_json")
    if capsule_data:
        st.download_button(
            label="Download omega_replay_capsule.json",
            data=capsule_data,
            file_name="omega_replay_capsule.json",
            mime="application/json",
        )

    uploaded = st.file_uploader("Import replay capsule", type=["json"])
    if uploaded is not None:
        try:
            raw = uploaded.getvalue().decode("utf-8", errors="replace")
            st.session_state.omega = load_replay_capsule(raw)
            st.success("Replay capsule loaded. Session state replaced.")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Failed to load capsule: {e}")

# Audit export and integrity
ops_exp = st.sidebar.expander("Audit / integrity", expanded=False)
with ops_exp:
    if st.button("Prepare audit log JSON"):
        st.session_state._audit_json = omega.audit.to_json()

    audit_json = st.session_state.get("_audit_json")
    if audit_json:
        st.download_button(
            label="Download omega_audit.json",
            data=audit_json,
            file_name="omega_audit.json",
            mime="application/json",
        )

    if st.button("Verify hash chains"):
        integrity = omega.verify_integrity()
        if integrity["audit_ok"] and integrity["memory_ok"]:
            st.success("Audit and memory hash chains verified OK.")
        else:
            msg_parts = []
            if not integrity["audit_ok"]:
                msg_parts.append(f"Audit chain broken at index {integrity['audit_first_bad_index']}.")
            if not integrity["memory_ok"]:
                msg_parts.append(f"Memory chain broken at index {integrity['memory_first_bad_index']}.")
            st.error(" ".join(msg_parts))

# --- main layout ----------------------------------------------------------
st.title("OmegaV4 – Unified Autonomous Oversight Kernel")
st.caption(
    "Multi-asset deterministic safety kernel · Δ-risk attribution · epistemic uncertainty gates · "
    "tamper-evident audit · replay capsule · human-gated autonomy."
)

scenario = st.text_area(
    "Describe the system / scenario OmegaV4 is supervising.",
    height=140,
    placeholder="Example: Supervise an autonomous mining haul truck fleet under human-gated control...",
)

top_buttons = st.columns([1, 1, 4])
with top_buttons[0]:
    run_tick = st.button("Advance Tick")
with top_buttons[1]:
    reset = st.button("Reset session")

if reset:
    for key in ["omega", "clarity_hist", "risk_hist", "pred_risk_hist", "epistemic_hist", "gate_pressure_hist"]:
        if key in st.session_state:
            del st.session_state[key]
    st.experimental_rerun()

# Ensure history dicts
for hist_key in ["clarity_hist", "risk_hist", "pred_risk_hist", "epistemic_hist", "gate_pressure_hist"]:
    if hist_key not in st.session_state:
        st.session_state[hist_key] = {}

tick_results: Dict[str, TickResult] = {}
tick_result: Optional[TickResult] = None

if run_tick and scenario.strip():
    tick_results = omega.tick_all(scenario.strip())

    # Update histories for all assets
    for key, res in tick_results.items():
        dec = res.avalon["decision"]
        scores = dec["scores"]

        st.session_state.clarity_hist.setdefault(key, []).append(float(scores["clarity"]))
        st.session_state.risk_hist.setdefault(key, []).append(float(scores["risk"]))
        st.session_state.pred_risk_hist.setdefault(key, []).append(float(dec["predicted_risk"]))
        st.session_state.epistemic_hist.setdefault(key, []).append(float(res.epistemic.epistemic))
        st.session_state.gate_pressure_hist.setdefault(key, []).append(float(res.decision.gate_pressure_index))

    tick_result = tick_results.get(active_asset_key)

# --- status bar -----------------------------------------------------------
snapshot = omega.snapshot_asset(active_asset_key)
if snapshot is None:
    st.error(f"Unknown asset {active_asset_key}")
    st.stop()

world_dict = snapshot["world"]
world_tick = snapshot["tick"]
world_mode = snapshot["mode"]
tick_ms_last = snapshot["tick_ms"]
vehicle = world_dict["vehicle"]

status_cols = st.columns(7)
with status_cols[0]:
    st.metric("Asset", active_asset_key)
with status_cols[1]:
    st.metric("Tick", world_tick)
with status_cols[2]:
    st.metric("Mode", world_mode.upper())
with status_cols[3]:
    st.metric("Drift (deg)", f"{vehicle['drift_deg']:.1f}")
with status_cols[4]:
    st.metric("Stability", f"{vehicle['stability']:.1f}")
with status_cols[5]:
    st.metric("Last tick (ms)", f"{tick_ms_last:.2f}")
with status_cols[6]:
    st.metric("Safety credit", f"{snapshot['safety_credit']:.1f}")

st.markdown("---")

# --- panels ---------------------------------------------------------------
top_l, top_r = st.columns([1.2, 1.3])

with top_l:
    st.markdown("### Safety Envelope + Epistemics")
    if tick_result is not None:
        risk = tick_result.risk
        decision = tick_result.decision
        epistemic = tick_result.epistemic

        st.metric("Risk band", risk.band)
        st.metric("Risk (0–1)", f"{risk.risk:.3f}")
        st.metric("Epistemic (0–1)", f"{epistemic.epistemic:.3f}")
        st.metric("Gate pressure", f"{decision.gate_pressure_index:.2f}")
        st.metric("Governor action", decision.action)

        if epistemic.knowledge_hole:
            st.warning("Knowledge hole detected: uncertainty exceeded threshold. Conservative gating is recommended.")

        if decision.requires_human_gate:
            st.warning(f"Human gate required for gate_id={decision.human_gate_id} (mode={world_mode}, band={decision.band}).")
        elif risk.band in ("HOLD", "STOP"):
            st.info("Envelope is in HOLD/STOP band but no LIVE gate required in this mode.")
        else:
            st.info("Risk is within configured envelopes for this mode.")
    else:
        st.info("Run at least one tick (with a scenario) to see safety envelope metrics.")

    st.markdown("#### Δ-Risk attribution (top drivers)")
    if tick_result is not None:
        topf = tick_result.risk.attribution.top_factors
        if topf:
            df_top = pd.DataFrame(topf)
            st.dataframe(df_top, use_container_width=True, height=180)
        else:
            st.caption("No attribution factors available.")
    else:
        st.caption("No tick yet.")

    st.markdown("#### Invariant proofs (LIVE enforced)")
    if tick_result is not None:
        proofs = [asdict(p) for p in tick_result.decision.invariant_proofs]
        st.dataframe(pd.DataFrame(proofs), use_container_width=True, height=220)
    else:
        st.caption("No tick yet.")

    st.markdown("#### World snapshot (active asset)")
    st.json(world_dict)

with top_r:
    st.markdown("### Avalon Oversight & Scores")
    if tick_result is not None:
        avalon_res = tick_result.avalon
        dec = avalon_res["decision"]
        scores = dec["scores"]

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Winning agent", dec["winner"] or "N/A")
        with m2:
            st.metric("Clarity (%)", f"{scores['clarity']:.1f}")
        with m3:
            st.metric("Text risk (%)", f"{scores['risk']:.1f}")
        with m4:
            st.metric("Pred. risk (%)", f"{dec['predicted_risk']:.1f}")
        with m5:
            st.metric("Epistemic", f"{tick_result.epistemic.epistemic:.2f}")

        if dec["predicted_risk"] >= risk_threshold:
            st.warning(f"Trajectory watch: predicted risk {dec['predicted_risk']:.1f}% ≥ threshold {risk_threshold}%.")
        elif scores["clarity"] < clarity_target:
            st.info(f"Clarity {scores['clarity']:.1f}% < target {clarity_target}%. Recommend additional human review or more data.")
        else:
            st.success("Clarity and predicted risk are inside configured envelopes.")

        with st.expander("Winning oversight item", expanded=True):
            item = dec.get("item") or {}
            st.markdown(f"**Agent:** {dec['winner'] or 'N/A'}")
            st.markdown(f"**Summary:** {item.get('summary', '')}")

            risks_list = item.get("risks") or []
            if risks_list:
                st.markdown("**Risks:**")
                for r in risks_list:
                    st.markdown(f"- {r}")

            recs_list = item.get("recommendations") or []
            if recs_list:
                st.markdown("**Recommendations:**")
                for r in recs_list:
                    st.markdown(f"- {r}")

            tags_list = item.get("tags") or []
            if tags_list:
                st.markdown(f"**Tags:** {', '.join(tags_list)}")

        st.markdown("#### All agent scores (this tick)")
        score_rows: List[Dict[str, Any]] = []
        for name, sc in avalon_res["scores"].items():
            row = {"Agent": name}
            row.update(sc)
            score_rows.append(row)

        if score_rows:
            df_scores = pd.DataFrame(score_rows).sort_values("overall", ascending=False)
            st.dataframe(df_scores, use_container_width=True, height=240)
        else:
            st.info("No agent scores to display.")
    else:
        st.info("Run at least one tick to see Avalon scores.")

st.markdown("---")

mid_l, mid_r = st.columns([1.4, 1.0])

with mid_l:
    st.markdown("### Trajectory – History (active asset)")
    if active_asset_key in st.session_state.clarity_hist and st.session_state.clarity_hist[active_asset_key]:
        hist_df = pd.DataFrame(
            {
                "step": list(range(1, len(st.session_state.clarity_hist[active_asset_key]) + 1)),
                "Avalon clarity": st.session_state.clarity_hist[active_asset_key],
                "Avalon text risk": st.session_state.risk_hist[active_asset_key],
                "Predicted risk": st.session_state.pred_risk_hist[active_asset_key],
                "Epistemic": st.session_state.epistemic_hist.get(active_asset_key, []),
                "Gate pressure": st.session_state.gate_pressure_hist.get(active_asset_key, []),
            }
        ).set_index("step")
        st.line_chart(hist_df)
    else:
        st.caption("No history yet for this asset. Run a few ticks with a scenario.")

    st.markdown("### Policy Shadow Diff (this tick)")
    if tick_result is not None and tick_result.shadow_diffs:
        df_sd = pd.DataFrame(tick_result.shadow_diffs).sort_values(["band_delta_rank", "gate_delta"], ascending=[False, False])
        st.dataframe(df_sd, use_container_width=True, height=220)
    else:
        st.caption("No policy shadow diffs yet. Run a tick.")

    st.markdown("### Memory Frames (tail, active asset)")
    mem_tail = omega.memory.tail(10, asset_key=active_asset_key)
    if mem_tail:
        mem_rows = [
            {
                "id": f.id,
                "tick": f.tick,
                "mode": f.mode,
                "band": f.risk_band,
                "risk": round(f.risk, 3),
                "epistemic": round(f.epistemic, 3),
                "gate_pressure": round(f.gate_pressure_index, 2),
                "action": f.action,
                "pred_risk": round(f.predicted_risk, 1),
                "winner": f.winner_agent,
                "gate_pending": f.human_gate_pending,
                "gate_id": f.human_gate_id,
                "hash": f.hash[:10] + "...",
            }
            for f in mem_tail
        ]
        st.dataframe(pd.DataFrame(mem_rows), use_container_width=True, height=260)
    else:
        st.caption("No memory frames yet for this asset.")

with mid_r:
    st.markdown("### Human Gate & Fleet Gates")

    if tick_result is not None:
        decision = tick_result.decision
        if decision.requires_human_gate and decision.human_gate_id:
            st.markdown(f"Pending gate for active asset: **{decision.human_gate_id}**")
            approve = st.checkbox("I approve the proposed envelope action.", value=False)
            note = st.text_input("Operator note (optional):")

            if st.button("Record gate decision"):
                ctx = omega.assets[active_asset_key]
                result = ctx.governor.apply_gate(
                    decision.human_gate_id,
                    approved=approve,
                    operator_id="operator_anon",
                    note=note,
                )
                omega.audit.log(
                    "human_gate_decision",
                    {
                        "asset": asdict(ctx.asset),
                        "tick": decision.tick,
                        "gate_id": decision.human_gate_id,
                        "approved": approve,
                        "note": note,
                        "result": result,
                    },
                )
                st.success("Decision recorded in audit log. OmegaV4 does not actuate; this is log-only.")
        else:
            st.info("No human gate required for the latest tick on this asset.")
    else:
        st.caption("Run a tick to see gates for the active asset.")

    st.markdown("#### Fleet gate queue (all assets)")
    gate_queue = omega.get_gate_queue()
    if gate_queue:
        gate_rows = [
            {
                "asset_key": g.asset_key,
                "gate_id": g.gate_id,
                "band": g.band,
                "mode": g.mode,
                "stage": g.stage,
                "severity": g.severity,
                "required_role": g.required_role,
                "created_at": g.created_at,
                "expires_at_tick": g.expires_at_tick,
            }
            for g in gate_queue
        ]
        st.dataframe(pd.DataFrame(gate_rows), use_container_width=True, height=220)
    else:
        st.caption("No pending gates across the fleet.")

st.markdown("---")

st.markdown("### Audit Trail (Recent Events)")
audit_tail = omega.audit.tail(20)
if audit_tail:
    rows = [
        {
            "seq": e.seq,
            "ts": e.timestamp,
            "kind": e.kind,
            "hash": e.hash[:12] + "...",
            "prev_hash": e.prev_hash[:12] + "...",
        }
        for e in audit_tail
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)
else:
    st.caption("No audit events yet.")

# --- Fleet interactions overview ------------------------------------------
st.markdown("### Fleet Interactions Overview")
fleet_rows: List[Dict[str, Any]] = []
for key, ctx in omega.assets.items():
    inter = omega.last_interactions.get(
        key,
        InteractionSummary(nearest_distance=MAX_PROX_DISTANCE, same_zone_conflict=False, convoy_spacing_bad=False),
    )
    fleet_rows.append(
        {
            "asset_key": key,
            "site": ctx.asset.site,
            "asset": ctx.asset.asset,
            "tick": ctx.world.tick,
            "zone": ctx.world.pose.zone,
            "x": round(ctx.world.pose.x, 2),
            "band": ctx.last_risk_band,
            "epistemic": round(ctx.last_epistemic, 3),
            "gate_pressure": round(ctx.last_gate_pressure, 2),
            "knowledge_hole": ctx.last_knowledge_hole,
            "nearest_distance": round(inter.nearest_distance, 2),
            "same_zone_conflict": inter.same_zone_conflict,
            "convoy_spacing_bad": inter.convoy_spacing_bad,
        }
    )

if fleet_rows:
    st.dataframe(pd.DataFrame(fleet_rows), use_container_width=True)
else:
    st.caption("No assets yet.")

st.caption(
    "OmegaV4 demo – single-file deterministic oversight kernel with Δ-risk attribution, epistemic uncertainty gating, "
    "policy shadow diffs, replay capsules, and human-in-the-loop guardrails. "
    "Extend by wiring in real telemetry and actuation layers while keeping the safety kernel, governor, audit spine, "
    "and human gate as non-negotiable rails."
)
