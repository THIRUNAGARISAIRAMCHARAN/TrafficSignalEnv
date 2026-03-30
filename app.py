from __future__ import annotations

from typing import Optional

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.models import Action, Observation
from env.tasks import TASK_CONFIGS, grade_episode, list_tasks
from env.traffic_env import TrafficEnvironment

app = FastAPI(title="TrafficSignalEnv", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = TrafficEnvironment()
current_task_id = "task1"
episode_data = {
    "steps": [],
    "total_reward": 0,
    "total_throughput": 0,
    "queue_samples": [],
    "emergency_cleared": False,
    "emergency_clear_time": None,
}


class ResetRequest(BaseModel):
    task_id: Optional[str] = "task1"


def _empty_episode_data() -> dict:
    return {
        "steps": [],
        "total_reward": 0,
        "total_throughput": 0,
        "queue_samples": [],
        "emergency_cleared": False,
        "emergency_clear_time": None,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "environment": "TrafficSignalEnv", "version": "1.0.0"}


@app.post("/reset")
def reset(body: Optional[ResetRequest] = Body(default=None)) -> dict:
    global current_task_id, episode_data

    episode_data = _empty_episode_data()
    if body is None:
        body = ResetRequest()
    current_task_id = body.task_id or "task1"

    config = TASK_CONFIGS.get(current_task_id, TASK_CONFIGS["task1"])
    obs: Observation = env.reset(task_config=config)
    return obs.model_dump()


@app.post("/step")
def step(body: Action) -> dict:
    global episode_data

    try:
        obs, reward, done, info = env.step(body)
    except IndexError as exc:
        raise HTTPException(status_code=400, detail="Invalid intersection_id") from exc

    episode_data["steps"].append(
        {
            "step": env.step_count,
            "reward": reward.value,
            "throughput": info.get("total_throughput", 0),
            "queued": info.get("total_waiting_cars", 0),
        }
    )
    episode_data["total_reward"] += reward.value
    episode_data["total_throughput"] = info.get("total_throughput", 0)
    episode_data["queue_samples"].append(info.get("total_waiting_cars", 0))

    if done:
        avg_queue = (
            sum(episode_data["queue_samples"]) / len(episode_data["queue_samples"])
            if episode_data["queue_samples"]
            else 0
        )
        episode_summary = {
            "total_throughput": episode_data["total_throughput"],
            "avg_queue_length": avg_queue,
            "emergency_cleared": episode_data["emergency_cleared"],
            "emergency_clear_time": episode_data["emergency_clear_time"]
            if episode_data["emergency_clear_time"] is not None
            else 999,
        }
        info["grade"] = grade_episode(current_task_id, episode_summary)

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict:
    return env.state().model_dump()


@app.get("/tasks")
def tasks() -> dict:
    return {"tasks": list_tasks()}


@app.post("/reset/{task_id}")
def reset_with_task(task_id: str) -> dict:
    global current_task_id, episode_data

    episode_data = _empty_episode_data()
    current_task_id = task_id
    config = TASK_CONFIGS.get(task_id, TASK_CONFIGS["task1"])
    obs: Observation = env.reset(task_config=config)
    return obs.model_dump()


def main() -> None:
    """Console entrypoint for OpenEnv / `[project.scripts]` (multi-mode deployment)."""
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
