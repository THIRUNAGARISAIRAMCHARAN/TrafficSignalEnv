from __future__ import annotations

from typing import Dict, List


TASK_CONFIGS: Dict[str, Dict] = {
    "task1": {
        "name": "Single Intersection Control",
        "difficulty": "easy",
        "num_intersections": 1,
        "max_steps": 300,
        "arrival_rate_ns": 0.3,
        "arrival_rate_ew": 0.3,
        "has_emergency": False,
        "emergency_step": None,
        "random_seed": 42,
        "target_throughput": 80,
        "max_avg_queue": 10,
    },
    "task2": {
        "name": "Multi-Intersection Coordination",
        "difficulty": "medium",
        "num_intersections": 4,
        "max_steps": 300,
        "arrival_rate_ns": 0.4,
        "arrival_rate_ew": 0.4,
        "has_emergency": False,
        "emergency_step": None,
        "random_seed": 123,
        "target_throughput": 280,
        "max_avg_queue": 12,
    },
    "task3": {
        "name": "Rush Hour Emergency Response",
        "difficulty": "hard",
        "num_intersections": 4,
        "max_steps": 300,
        "arrival_rate_ns": 0.7,
        "arrival_rate_ew": 0.5,
        "has_emergency": True,
        "emergency_step": 100,
        "random_seed": 999,
        "target_throughput": 350,
        "max_avg_queue": 15,
    },
}


def grade_episode(task_id: str, episode_info: dict) -> float:
    config = TASK_CONFIGS[task_id]
    total_tp = episode_info.get("total_throughput", 0)
    avg_q = episode_info.get("avg_queue_length", 999)
    emerg_cleared = episode_info.get("emergency_cleared", False)
    emerg_time = episode_info.get("emergency_clear_time", 999)

    throughput_score = min(total_tp / config["target_throughput"], 1.0) * 0.5
    queue_score = max(0.0, 1.0 - avg_q / config["max_avg_queue"])

    if config["has_emergency"]:
        queue_score *= 0.3
        emerg_score = 0.2 if emerg_cleared and emerg_time <= 30 else 0.0
    else:
        queue_score *= 0.5
        emerg_score = 0.0

    return round(throughput_score + queue_score + emerg_score, 4)


def get_task_config(task_id: str) -> dict:
    return TASK_CONFIGS[task_id]


def list_tasks() -> List[dict]:
    return [{"id": k, **{f: v for f, v in c.items()}} for k, c in TASK_CONFIGS.items()]
