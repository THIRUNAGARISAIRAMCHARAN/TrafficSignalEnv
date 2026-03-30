---
title: TrafficSignalEnv
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
pinned: true
tags:
  - openenv
  - reinforcement-learning
  - traffic
  - optimization
---

# 🚦 TrafficSignalEnv

TrafficSignalEnv is an OpenEnv-style environment for training and evaluating agents on traffic signal control. The agent observes queue lengths, signal phases, and emergency status across up to four intersections, then issues commands—extend green, switch phase, or emergency override—to move vehicles through the network while balancing throughput, waiting, and rush-hour emergencies.

## Environment Description

The simulation uses **Poisson arrivals** to add vehicles to north–south and east–west queues each step. Each intersection cycles through **five signal phases** in order: **NS_GREEN**, **NS_YELLOW**, **ALL_RED**, **EW_GREEN**, **EW_YELLOW**, then **ALL_RED** again, with configurable dwell times per phase. While **NS_GREEN** or **EW_GREEN** is active, up to **two cars per second** (per timestep with `dt=1`) may clear from the corresponding queue and count toward throughput. The environment supports **up to four intersections** and runs **300 steps** per episode by default (`max_steps`).

## Action Space

| Command | Description | Parameter |
|---------|-------------|-----------|
| `extend_green` | Extend current green phase | `duration` (sec) |
| `switch_phase` | Force next phase transition | none |
| `emergency_override` | Force NS green 20 sec | none |

## Observation Space

Top-level **Observation** fields:

| Field | Type | Description |
|-------|------|-------------|
| `intersections` | array | List of per-intersection state objects |
| `step_count` | integer | Current timestep in the episode |
| `elapsed_time` | float | Simulated seconds elapsed |
| `total_waiting_cars` | integer | Cars waiting summed over all intersections |
| `emergency_active` | boolean | Whether an emergency is active anywhere |

Each element of `intersections` (**IntersectionState**) includes:

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Intersection index |
| `current_phase` | string (enum) | Active signal phase |
| `phase_timer` | float | Time spent in the current phase |
| `ns_queue` | integer | North–south waiting vehicles |
| `ew_queue` | integer | East–west waiting vehicles |
| `ns_throughput` | integer | Cumulative NS vehicles cleared |
| `ew_throughput` | integer | Cumulative EW vehicles cleared |
| `has_emergency` | boolean | Emergency flag for this intersection |

## Tasks

| Task ID | Name | Difficulty | Summary |
|---------|------|------------|---------|
| task1 | Single Intersection | easy | 1 intersection, 300 steps |
| task2 | Multi-Intersection | medium | 4 intersections, 300 steps |
| task3 | Rush Hour Emergency | hard | 4 intersections + emergency |

## Reward Function

Each step, the **Reward** model breaks down the scalar `value` into:

- **throughput_bonus** — Positive contribution proportional to cars cleared this step (e.g. 0.5 per car cleared in the implementation).
- **wait_penalty** — Negative contribution from total queued cars across intersections (waiting imposes a per-car penalty).
- **emergency_bonus** — Bonus when an emergency has just been cleared; small negative while an emergency remains active, encouraging timely handling.
- **starvation_penalty** — Extra penalty when any lane queue grows very large (e.g. over a threshold), discouraging neglected directions.

The step reward `value` is the sum of these components.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn app:app --port 7860
curl -X POST localhost:7860/reset -d '{}'
```

## Docker

```bash
docker build -t traffic-env .
docker run -p 7860:7860 traffic-env
```

## Baseline Scores

| Task | Grade |
|------|-------|
| task1 (easy) | ~0.62 |
| task2 (medium) | ~0.51 |
| task3 (hard) | ~0.38 |
