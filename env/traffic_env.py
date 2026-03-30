from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from env.models import Action, IntersectionState, Observation, Reward, TrafficPhase


class TrafficIntersection:
    def __init__(self, id, arrival_rate_ns=0.3, arrival_rate_ew=0.3, seed=None):
        self.id = id
        self.current_phase = TrafficPhase.NS_GREEN
        self.phase_timer = 0.0
        self.ns_queue = 0
        self.ew_queue = 0
        self.ns_throughput = 0
        self.ew_throughput = 0
        self.has_emergency = False
        self.arrival_rate_ns = arrival_rate_ns
        self.arrival_rate_ew = arrival_rate_ew

        self.phase_durations = {
            TrafficPhase.NS_GREEN: 30,
            TrafficPhase.EW_GREEN: 30,
            TrafficPhase.NS_YELLOW: 4,
            TrafficPhase.EW_YELLOW: 4,
            TrafficPhase.ALL_RED: 2,
        }
        self.phase_sequence = [
            TrafficPhase.NS_GREEN,
            TrafficPhase.NS_YELLOW,
            TrafficPhase.ALL_RED,
            TrafficPhase.EW_GREEN,
            TrafficPhase.EW_YELLOW,
            TrafficPhase.ALL_RED,
        ]

        self._rng = np.random.default_rng(seed)
        self._last_cleared_ns = 0
        self._last_cleared_ew = 0
        self._last_emergency_active = False
        self._emergency_just_cleared = False

    def _advance_phase(self) -> None:
        current_index = self.phase_sequence.index(self.current_phase)
        next_index = (current_index + 1) % len(self.phase_sequence)
        self.current_phase = self.phase_sequence[next_index]
        self.phase_timer = 0.0

    def step(self, dt=1.0) -> Tuple[int, int]:
        self.ns_queue += int(self._rng.poisson(self.arrival_rate_ns * dt))
        self.ew_queue += int(self._rng.poisson(self.arrival_rate_ew * dt))

        cars_cleared_ns = 0
        cars_cleared_ew = 0

        if self.current_phase == TrafficPhase.NS_GREEN:
            cars_cleared_ns = min(self.ns_queue, 2)
            self.ns_queue -= cars_cleared_ns
            self.ns_throughput += cars_cleared_ns
        elif self.current_phase == TrafficPhase.EW_GREEN:
            cars_cleared_ew = min(self.ew_queue, 2)
            self.ew_queue -= cars_cleared_ew
            self.ew_throughput += cars_cleared_ew

        self.phase_timer += dt

        if self.phase_timer >= self.phase_durations[self.current_phase]:
            self._advance_phase()

        self._last_cleared_ns = cars_cleared_ns
        self._last_cleared_ew = cars_cleared_ew
        self._emergency_just_cleared = self._last_emergency_active and not self.has_emergency
        self._last_emergency_active = self.has_emergency

        return cars_cleared_ns, cars_cleared_ew

    def apply_action(self, command, duration=None):
        if command == "extend_green":
            if self.current_phase in (TrafficPhase.NS_GREEN, TrafficPhase.EW_GREEN):
                extension = min(duration or 10, 15)
                self.phase_timer -= extension
                if self.phase_timer < 0:
                    self.phase_timer = 0.0
        elif command == "switch_phase":
            self._advance_phase()
        elif command == "emergency_override":
            self.current_phase = TrafficPhase.NS_GREEN
            self.phase_timer = 0.0
            self.phase_durations[TrafficPhase.NS_GREEN] = 20
            self.has_emergency = True

    def get_state(self) -> IntersectionState:
        return IntersectionState(
            id=self.id,
            current_phase=self.current_phase,
            phase_timer=self.phase_timer,
            ns_queue=self.ns_queue,
            ew_queue=self.ew_queue,
            ns_throughput=self.ns_throughput,
            ew_throughput=self.ew_throughput,
            has_emergency=self.has_emergency,
        )


class TrafficEnvironment:
    def __init__(self, num_intersections=1, task_config=None):
        self.num_intersections = num_intersections
        self.task_config = task_config or {}
        self.intersections: List[TrafficIntersection] = []
        self.step_count = 0
        self.elapsed_time = 0.0
        self.max_steps = 300
        self.episode_info: Dict = {}
        self._last_total_cleared = 0

        self.reset(self.task_config)

    def reset(self, task_config=None) -> Observation:
        self.task_config = task_config or self.task_config or {}
        self.intersections = []

        for i in range(self.num_intersections):
            intersection_cfg = self.task_config.get("intersections", {}).get(i, {})
            arrival_rate_ns = intersection_cfg.get("arrival_rate_ns", self.task_config.get("arrival_rate_ns", 0.3))
            arrival_rate_ew = intersection_cfg.get("arrival_rate_ew", self.task_config.get("arrival_rate_ew", 0.3))
            seed = intersection_cfg.get("seed", self.task_config.get("seed"))
            self.intersections.append(
                TrafficIntersection(
                    id=i,
                    arrival_rate_ns=arrival_rate_ns,
                    arrival_rate_ew=arrival_rate_ew,
                    seed=seed,
                )
            )

        self.step_count = 0
        self.elapsed_time = 0.0
        self.episode_info = {}
        self._last_total_cleared = 0
        return self.state()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, dict]:
        self.intersections[action.intersection_id].apply_action(action.command, action.duration)

        total_cleared = 0
        for intersection in self.intersections:
            cleared_ns, cleared_ew = intersection.step(1.0)
            total_cleared += cleared_ns + cleared_ew

        self._last_total_cleared = total_cleared
        self.step_count += 1
        self.elapsed_time += 1.0

        reward = self._compute_reward()
        done = self._check_done()

        total_waiting_cars = sum(i.ns_queue + i.ew_queue for i in self.intersections)
        total_throughput = sum(i.ns_throughput + i.ew_throughput for i in self.intersections)
        info = {
            "step_count": self.step_count,
            "elapsed_time": self.elapsed_time,
            "total_waiting_cars": total_waiting_cars,
            "total_throughput": total_throughput,
            "max_steps": self.max_steps,
        }
        self.episode_info = info

        return self.state(), reward, done, info

    def state(self) -> Observation:
        intersections = [intersection.get_state() for intersection in self.intersections]
        total_waiting_cars = sum(s.ns_queue + s.ew_queue for s in intersections)
        emergency_active = any(s.has_emergency for s in intersections)
        return Observation(
            intersections=intersections,
            step_count=self.step_count,
            elapsed_time=self.elapsed_time,
            total_waiting_cars=total_waiting_cars,
            emergency_active=emergency_active,
        )

    def _compute_reward(self) -> Reward:
        total_cleared = self._last_total_cleared
        total_queued = sum(i.ns_queue + i.ew_queue for i in self.intersections)

        throughput_bonus = total_cleared * 0.5
        wait_penalty = total_queued * -0.1
        starvation_penalty = sum(
            -5
            for i in self.intersections
            for q in (i.ns_queue, i.ew_queue)
            if q > 20
        )

        emergency_just_cleared = any(i._emergency_just_cleared for i in self.intersections)
        emergency_active = any(i.has_emergency for i in self.intersections)
        if emergency_just_cleared:
            emergency_bonus = 50
        elif emergency_active:
            emergency_bonus = -1
        else:
            emergency_bonus = 0

        value = throughput_bonus + wait_penalty + starvation_penalty + emergency_bonus
        return Reward(
            value=value,
            throughput_bonus=throughput_bonus,
            wait_penalty=wait_penalty,
            emergency_bonus=emergency_bonus,
            starvation_penalty=starvation_penalty,
        )

    def _check_done(self) -> bool:
        return self.step_count >= self.max_steps
