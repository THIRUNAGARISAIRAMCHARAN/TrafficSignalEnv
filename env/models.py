from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class TrafficPhase(str, Enum):
    NS_GREEN = "NS_GREEN"
    EW_GREEN = "EW_GREEN"
    NS_YELLOW = "NS_YELLOW"
    EW_YELLOW = "EW_YELLOW"
    ALL_RED = "ALL_RED"


class IntersectionState(BaseModel):
    id: int
    current_phase: TrafficPhase
    phase_timer: float
    ns_queue: int
    ew_queue: int
    ns_throughput: int
    ew_throughput: int
    has_emergency: bool


class Observation(BaseModel):
    intersections: List[IntersectionState]
    step_count: int
    elapsed_time: float
    total_waiting_cars: int
    emergency_active: bool


class Action(BaseModel):
    intersection_id: int
    command: str  # extend_green, switch_phase, emergency_override
    duration: Optional[float] = None


class Reward(BaseModel):
    value: float
    throughput_bonus: float
    wait_penalty: float
    emergency_bonus: float
    starvation_penalty: float
