"""Explicit, testable circuit-breaker state for GPU health signals.

This private prototype separates observation from state transition: status reads
never mutate the breaker. Recovery probes are opened explicitly after timeout.
No CoreWeave production behavior or reliability improvement is claimed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable


class CircuitState:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True, slots=True)
class BreakerStatus:
    state: str
    failures: int
    can_execute: bool
    recovery_probe_ready: bool


class GPUCircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        recovery_timeout: float = 300.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be positive")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be positive")
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._clock = clock
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time: float | None = None

    def record_success(self) -> None:
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
            return
        if self.state == CircuitState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = self._clock()
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            return
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def recovery_probe_ready(self) -> bool:
        if self.state != CircuitState.OPEN or self.last_failure_time is None:
            return False
        return self._clock() - self.last_failure_time >= self.recovery_timeout

    def open_recovery_probe(self) -> bool:
        if not self.recovery_probe_ready():
            return False
        self.state = CircuitState.HALF_OPEN
        return True

    def can_execute(self) -> bool:
        return self.state in {CircuitState.CLOSED, CircuitState.HALF_OPEN}

    def get_status(self) -> BreakerStatus:
        return BreakerStatus(
            state=self.state,
            failures=self.failure_count,
            can_execute=self.can_execute(),
            recovery_probe_ready=self.recovery_probe_ready(),
        )


class ClusterCircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        recovery_timeout: float = 300.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._clock = clock
        self.gpu_breakers: dict[int, GPUCircuitBreaker] = {}
        self._isolation_log: list[dict[str, object]] = []

    def breaker_for(self, gpu_id: int) -> GPUCircuitBreaker:
        if gpu_id < 0:
            raise ValueError("gpu_id cannot be negative")
        if gpu_id not in self.gpu_breakers:
            self.gpu_breakers[gpu_id] = GPUCircuitBreaker(
                failure_threshold=self.failure_threshold,
                recovery_timeout=self.recovery_timeout,
                clock=self._clock,
            )
        return self.gpu_breakers[gpu_id]

    def record_health(self, gpu_id: int, *, healthy: bool) -> BreakerStatus:
        breaker = self.breaker_for(gpu_id)
        was_open = breaker.state == CircuitState.OPEN
        if healthy:
            breaker.record_success()
        else:
            breaker.record_failure()
        if not was_open and breaker.state == CircuitState.OPEN:
            self._isolation_log.append(
                {
                    "gpu_id": gpu_id,
                    "timestamp": self._clock(),
                    "failures": breaker.failure_count,
                }
            )
        return breaker.get_status()

    def open_recovery_probe(self, gpu_id: int) -> BreakerStatus:
        breaker = self.breaker_for(gpu_id)
        breaker.open_recovery_probe()
        return breaker.get_status()

    def get_cluster_status(self) -> dict[str, object]:
        states = {
            gpu_id: breaker.get_status().state
            for gpu_id, breaker in sorted(self.gpu_breakers.items())
        }
        return {
            "total_gpus": len(states),
            "closed": sum(state == CircuitState.CLOSED for state in states.values()),
            "open": sum(state == CircuitState.OPEN for state in states.values()),
            "half_open": sum(
                state == CircuitState.HALF_OPEN for state in states.values()
            ),
            "states": states,
            "isolation_events": list(self._isolation_log),
        }


if __name__ == "__main__":
    now = [1_000_000.0]
    cluster = ClusterCircuitBreaker(
        failure_threshold=3,
        recovery_timeout=30.0,
        clock=lambda: now[0],
    )
    for _ in range(3):
        cluster.record_health(3, healthy=False)
    print(json.dumps(cluster.get_cluster_status(), indent=2))
    now[0] += 31.0
    print(cluster.open_recovery_probe(3))
    cluster.record_health(3, healthy=True)
    print(json.dumps(cluster.get_cluster_status(), indent=2))
