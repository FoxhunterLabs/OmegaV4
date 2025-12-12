# OmegaV4 – Unified Autonomous Oversight Kernel

OmegaV4 is a **deterministic, non‑actuating oversight system** for autonomous and semi‑autonomous operations. It exists to **limit autonomy**, not expand it — enforcing contextual safety, epistemic humility, and human authority through inspectable logic, replayable state, and tamper‑evident audit.

> **Omega never actuates. Humans remain the final authority.**

---

## Why OmegaV4 Exists

Most autonomy stacks optimize capability first and bolt on safety later. OmegaV4 inverts that model.

OmegaV4:

* Supervises autonomy without controlling it
* Detects risk *and* ignorance
* Regulates human cognitive load
* Preserves accountability through audit and replay
* Prevents autonomy from scaling faster than responsibility

It is designed for **infrastructure, mining, heavy equipment, fleet ops, and defense‑adjacent systems** where human life, liability, and trust dominate performance metrics.

---

## Core Principles

* **Determinism first** – Every decision is replayable from seed + state
* **Oversight, not actuation** – Omega proposes, never commands
* **Human‑gated authority** – Explicit approval for high‑risk states
* **Epistemic humility** – “Unknown” is treated as a first‑class safety condition
* **Auditability** – All events are chained, verifiable, and exportable

---

## What’s New in V4

### 1. Counterfactual Δ‑Risk Attribution

OmegaV4 explains *why* risk changed by computing per‑factor deltas:

* Drift, stability, speed
* Sensor health, staleness, contradiction
* Proximity and zone conflicts
* Historical risk bias

This replaces opaque scores with **inspectable causality**.

---

### 2. Epistemic Uncertainty & Knowledge Holes

Risk and knowledge are separated.

OmegaV4 detects when the system **does not know enough to judge safety**, even if risk appears low. High epistemic uncertainty:

* Forces conservative behavior
* Triggers human review
* Can block autonomy despite low kinematic risk

---

### 3. Formal Invariant Proof Objects

Instead of simple violations, OmegaV4 emits proofs:

* Safety margin
* Confidence (adjusted by epistemic state)
* Evidence hashes

This enables post‑incident reasoning and external review.

---

### 4. Human Gate Load Regulation

Human operators are treated as a safety‑critical resource.

OmegaV4:

* Tracks gate frequency and escalation churn
* Applies cooldowns to prevent alert fatigue
* Can recommend supervision withdrawal if cognitive load becomes unsafe

---

### 5. Policy Shadowing & Diffing

Multiple policies run in parallel:

* Active policy
* Conservative shadow
* Permissive shadow

OmegaV4 reports how alternative policies *would have behaved* — without changing live behavior.

---

### 6. Deterministic Replay Capsules

Every session can be exported as a **single replay capsule** containing:

* Seeds and world state
* Policy and invariants
* Audit chain
* Memory chain

Capsules replay deterministically for audits, incidents, or review boards.

---

## System Architecture (High Level)

```
World & Sensors
        ↓
Safety Kernel (Risk + Epistemic)
        ↓
Invariant Proofs
        ↓
Governor (Modes + Human Gates)
        ↓
Avalon Oversight (Structured Agents)
        ↓
Memory Frames + Audit Spine
```

OmegaV4 can supervise **multiple assets simultaneously** while reasoning about proximity, zones, and convoy spacing.

---

## Modes of Operation

* **SHADOW** – Observe and log only (no gating)
* **TRAINING** – Gating simulated, no enforcement
* **LIVE** – Invariants enforced, human gates required

Mode changes are audited and reversible.

---

## What OmegaV4 Is *Not*

* ❌ An autonomy controller
* ❌ A planner or optimizer
* ❌ A black‑box ML system
* ❌ A replacement for operators

OmegaV4 exists to **slow systems down when needed**, not push them faster.

---

## Running the App

```bash
pip install streamlit pandas
streamlit run app.py
```

The Streamlit UI provides:

* Live risk and epistemic state
* Human gate queue
* Policy inspection
* Audit and memory verification
* Replay capsule export/import

---

## Intended Use Cases

* Autonomous mining haul trucks
* Heavy equipment fleets
* Infrastructure robotics
* Safety‑critical autonomy programs
* Human‑in‑the‑loop certification environments

---

## Design Philosophy

> **“Autonomy must never scale faster than accountability.”**

OmegaV4 is an anti‑autonomy system by design — a safety shell that protects people, operators, and organizations from overconfidence, incomplete knowledge, and institutional pressure.

---

## Status

OmegaV4 is a **single‑file reference implementation** intended to demonstrate architecture, not production readiness. It is designed to be inspected, extended, and debated.

---

## License & Ethics

Use OmegaV4 to **protect human life and responsibility**. Do not use it to remove humans from moral decision loops or to justify unsafe automation.

---

*OmegaV4 — Deterministic oversight for systems that matter.*
