"""GPU circuit breaking with explicit isolation and controlled recovery probes."""

from __future__ import annotations

import argparse
import json
import time
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class GPUCircuitBreaker:
    """Isolate a failing GPU and allow one controlled recovery probe at a time."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 300.0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if recovery_timeout < 0:
            raise ValueError("recovery_timeout must be >= 0")
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.opened_at: float | None = None
        self._probe_in_flight = False
        self._clock = clock

    def _now(self, now: float | None) -> float:
        return self._clock() if now is None else now

    def record_success(self, now: float | None = None) -> None:
        self._now(now)
        if self.state is CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.opened_at = None
            self._probe_in_flight = False
        elif self.state is CircuitState.CLOSED:
            self.failure_count = 0

    def record_failure(self, now: float | None = None) -> bool:
        current = self._now(now)
        was_open = self.state is CircuitState.OPEN
        self.failure_count += 1
        if self.state is CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.opened_at = current
            self._probe_in_flight = False
        return not was_open and self.state is CircuitState.OPEN

    def recovery_ready(self, now: float | None = None) -> bool:
        if self.state is not CircuitState.OPEN or self.opened_at is None:
            return False
        return self._now(now) - self.opened_at >= self.recovery_timeout

    def can_execute(self, now: float | None = None) -> bool:
        """Pure readiness query. It never transitions breaker state."""
        if self.state is CircuitState.CLOSED:
            return True
        if self.state is CircuitState.OPEN:
            return self.recovery_ready(now)
        return not self._probe_in_flight

    def acquire(self, now: float | None = None) -> bool:
        """Acquire execution permission, transitioning a ready OPEN breaker to HALF_OPEN."""
        if self.state is CircuitState.CLOSED:
            return True
        if self.state is CircuitState.OPEN:
            if not self.recovery_ready(now):
                return False
            self.state = CircuitState.HALF_OPEN
            self._probe_in_flight = True
            return True
        if self._probe_in_flight:
            return False
        self._probe_in_flight = True
        return True

    def status(self, now: float | None = None) -> dict[str, Any]:
        current = self._now(now)
        wait_s = 0.0
        if self.state is CircuitState.OPEN and self.opened_at is not None:
            wait_s = max(0.0, self.recovery_timeout - (current - self.opened_at))
        return {
            "state": self.state.value,
            "failures": self.failure_count,
            "can_execute": self.can_execute(current),
            "recovery_ready": self.recovery_ready(current),
            "seconds_until_probe": round(wait_s, 6),
            "probe_in_flight": self._probe_in_flight,
        }


class ClusterCircuitBreaker:
    """Manage independently isolated GPU circuits across a cluster."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 300.0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._clock = clock
        self.gpu_breakers: dict[str, GPUCircuitBreaker] = {}
        self._isolation_log: list[dict[str, Any]] = []
        self._recovery_log: list[dict[str, Any]] = []

    def breaker_for(self, gpu_id: str | int) -> GPUCircuitBreaker:
        key = str(gpu_id)
        if key not in self.gpu_breakers:
            self.gpu_breakers[key] = GPUCircuitBreaker(
                failure_threshold=self.failure_threshold,
                recovery_timeout=self.recovery_timeout,
                clock=self._clock,
            )
        return self.gpu_breakers[key]

    def record_health(
        self, gpu_id: str | int, healthy: bool, now: float | None = None
    ) -> dict[str, Any]:
        current = self._clock() if now is None else now
        breaker = self.breaker_for(gpu_id)
        if healthy:
            breaker.record_success(current)
        else:
            newly_opened = breaker.record_failure(current)
            if newly_opened:
                self._isolation_log.append(
                    {"gpu_id": str(gpu_id), "opened_at": current, "reason": "failure_threshold"}
                )
        return {"gpu_id": str(gpu_id), **breaker.status(current)}

    def acquire_gpu(self, gpu_id: str | int, now: float | None = None) -> bool:
        current = self._clock() if now is None else now
        breaker = self.breaker_for(gpu_id)
        previous = breaker.state
        allowed = breaker.acquire(current)
        if allowed and previous is CircuitState.OPEN:
            self._recovery_log.append(
                {"gpu_id": str(gpu_id), "probe_started_at": current}
            )
        return allowed

    def get_cluster_status(self, now: float | None = None) -> dict[str, Any]:
        current = self._clock() if now is None else now
        states = {state.value: 0 for state in CircuitState}
        for breaker in self.gpu_breakers.values():
            states[breaker.state.value] += 1
        return {
            "total_gpus": len(self.gpu_breakers),
            "states": states,
            "schedulable": sum(
                1 for breaker in self.gpu_breakers.values() if breaker.can_execute(current)
            ),
            "isolations": len(self._isolation_log),
            "recovery_probes": len(self._recovery_log),
        }

    def receipt(self, now: float | None = None) -> dict[str, Any]:
        current = self._clock() if now is None else now
        return {
            "cluster": self.get_cluster_status(current),
            "gpus": {
                gpu_id: breaker.status(current)
                for gpu_id, breaker in sorted(self.gpu_breakers.items())
            },
            "isolation_log": list(self._isolation_log),
            "recovery_log": list(self._recovery_log),
        }


def run_events(
    events: list[dict[str, Any]], *, failure_threshold: int = 3, recovery_timeout: float = 300.0
) -> dict[str, Any]:
    cluster = ClusterCircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
    )
    last_time = 0.0
    for event in events:
        event_type = str(event.get("type", "health"))
        gpu_id = event["gpu_id"]
        at = float(event.get("at", last_time))
        last_time = max(last_time, at)
        if event_type == "health":
            cluster.record_health(gpu_id, bool(event["healthy"]), now=at)
        elif event_type == "probe":
            allowed = cluster.acquire_gpu(gpu_id, now=at)
            if allowed and "healthy" in event:
                cluster.record_health(gpu_id, bool(event["healthy"]), now=at)
        else:
            raise ValueError(f"unknown event type: {event_type}")
    return cluster.receipt(last_time)


def demo() -> dict[str, Any]:
    return run_events(
        [
            {"type": "health", "gpu_id": "gpu-3", "healthy": False, "at": 0},
            {"type": "health", "gpu_id": "gpu-3", "healthy": False, "at": 1},
            {"type": "health", "gpu_id": "gpu-3", "healthy": False, "at": 2},
            {"type": "probe", "gpu_id": "gpu-3", "at": 7},
            {"type": "probe", "gpu_id": "gpu-3", "healthy": True, "at": 13},
            {"type": "health", "gpu_id": "gpu-4", "healthy": True, "at": 13},
        ],
        recovery_timeout=10,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPU circuit isolation and recovery")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("demo", help="Run deterministic isolation/recovery demo")
    run = subparsers.add_parser("run", help="Process JSON health/probe events")
    run.add_argument("--events", type=Path, required=True)
    run.add_argument("--failure-threshold", type=int, default=3)
    run.add_argument("--recovery-timeout", type=float, default=300.0)
    args = parser.parse_args(argv)

    if args.command == "demo":
        payload = demo()
    else:
        events = json.loads(args.events.read_text(encoding="utf-8"))
        if not isinstance(events, list):
            raise ValueError("events file must contain a JSON array")
        payload = run_events(
            events,
            failure_threshold=args.failure_threshold,
            recovery_timeout=args.recovery_timeout,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
