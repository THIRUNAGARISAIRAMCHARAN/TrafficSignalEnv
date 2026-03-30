from __future__ import annotations

import json
import os
import time

import httpx
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
MODEL = "gpt-4o-mini"
client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """
You are a traffic signal controller AI.
Given the current state of intersections, choose the best action.
Respond ONLY with valid JSON in this exact format:
{"intersection_id": 0, "command": "switch_phase", "duration": null}
Commands: extend_green (add green time), switch_phase (change signal),
emergency_override (give NS green for emergency vehicle).
Pick the intersection with the longest queue.
Use emergency_override if emergency_active is true.
"""


def format_observation(obs: dict) -> str:
    intersections = obs.get("intersections", [])
    step_count = obs.get("step_count", 0)
    elapsed_time = obs.get("elapsed_time", 0.0)
    total_waiting = obs.get("total_waiting_cars", 0)
    emergency_active = obs.get("emergency_active", False)

    lines = [
        f"step_count={step_count}",
        f"elapsed_time={elapsed_time}",
        f"total_waiting_cars={total_waiting}",
        f"emergency_active={emergency_active}",
        "intersections:",
    ]

    for item in intersections:
        lines.append(
            "- id={id}, phase={phase}, timer={timer:.1f}, ns_queue={ns_q}, ew_queue={ew_q}, ns_throughput={ns_tp}, ew_throughput={ew_tp}".format(
                id=item.get("id", 0),
                phase=item.get("current_phase", "UNKNOWN"),
                timer=float(item.get("phase_timer", 0.0)),
                ns_q=item.get("ns_queue", 0),
                ew_q=item.get("ew_queue", 0),
                ns_tp=item.get("ns_throughput", 0),
                ew_tp=item.get("ew_throughput", 0),
            )
        )

    return "\n".join(lines)


def _fallback_action() -> dict:
    return {"intersection_id": 0, "command": "switch_phase", "duration": None}


def get_action(obs: dict) -> dict:
    try:
        user_prompt = format_observation(obs)
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content if response.choices else ""
        action = json.loads(content or "")

        if not isinstance(action, dict):
            return _fallback_action()
        if "intersection_id" not in action or "command" not in action:
            return _fallback_action()

        return {
            "intersection_id": int(action.get("intersection_id", 0)),
            "command": str(action.get("command", "switch_phase")),
            "duration": action.get("duration", None),
        }
    except Exception:
        return _fallback_action()


def run_task(task_id: str) -> dict:
    total_reward = 0.0
    steps = 0
    grade = 0.0

    with httpx.Client(timeout=20.0) as http:
        reset_resp = http.post(f"{ENV_URL}/reset", json={"task_id": task_id})
        reset_resp.raise_for_status()
        obs = reset_resp.json()

        done = False
        for _ in range(310):
            if done:
                break

            action = get_action(obs)
            step_resp = http.post(f"{ENV_URL}/step", json=action)
            step_resp.raise_for_status()
            result = step_resp.json()

            obs = result.get("observation", obs)
            reward = result.get("reward", {})
            done = bool(result.get("done", False))
            info = result.get("info", {})

            total_reward += float(reward.get("value", 0.0))
            steps += 1

            if done:
                grade = float(info.get("grade", 0.0))

            time.sleep(0.01)

    return {
        "task_id": task_id,
        "steps": steps,
        "total_reward": total_reward,
        "grade": grade,
    }


def main():
    print("TrafficSignalEnv Baseline Evaluation")
    print()

    results = [run_task(t) for t in ["task1", "task2", "task3"]]

    print("Task       | Steps | Reward  | Grade")
    for r in results:
        print(
            f"{r['task_id']:<10} | {r['steps']:>5} | {r['total_reward']:>7.1f} | {r['grade']:.4f}"
        )

    avg_grade = sum(r["grade"] for r in results) / len(results) if results else 0.0
    print()
    print(f"Average Grade: {avg_grade:.4f}")
    return results


if __name__ == "__main__":
    main()
