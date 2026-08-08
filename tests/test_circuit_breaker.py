from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from circuit_breaker import (  # noqa: E402
    CircuitState,
    ClusterCircuitBreaker,
    GPUCircuitBreaker,
)


class FakeClock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class CircuitBreakerTests(unittest.TestCase):
    def test_three_failures_open_breaker(self) -> None:
        clock = FakeClock()
        breaker = GPUCircuitBreaker(clock=clock)

        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()

        self.assertEqual(breaker.state, CircuitState.OPEN)
        self.assertFalse(breaker.can_execute())

    def test_status_read_does_not_mutate_open_state(self) -> None:
        clock = FakeClock()
        breaker = GPUCircuitBreaker(
            failure_threshold=1,
            recovery_timeout=10.0,
            clock=clock,
        )
        breaker.record_failure()
        clock.now += 11.0

        status = breaker.get_status()

        self.assertEqual(status.state, CircuitState.OPEN)
        self.assertTrue(status.recovery_probe_ready)
        self.assertEqual(breaker.state, CircuitState.OPEN)

    def test_recovery_probe_requires_explicit_transition(self) -> None:
        clock = FakeClock()
        breaker = GPUCircuitBreaker(
            failure_threshold=1,
            recovery_timeout=10.0,
            clock=clock,
        )
        breaker.record_failure()
        self.assertFalse(breaker.open_recovery_probe())
        clock.now += 11.0

        self.assertTrue(breaker.open_recovery_probe())
        self.assertEqual(breaker.state, CircuitState.HALF_OPEN)
        breaker.record_success()
        self.assertEqual(breaker.state, CircuitState.CLOSED)
        self.assertEqual(breaker.failure_count, 0)

    def test_failed_half_open_probe_reopens_breaker(self) -> None:
        clock = FakeClock()
        breaker = GPUCircuitBreaker(
            failure_threshold=1,
            recovery_timeout=10.0,
            clock=clock,
        )
        breaker.record_failure()
        clock.now += 11.0
        breaker.open_recovery_probe()

        breaker.record_failure()

        self.assertEqual(breaker.state, CircuitState.OPEN)

    def test_cluster_records_one_isolation_transition(self) -> None:
        clock = FakeClock()
        cluster = ClusterCircuitBreaker(clock=clock)
        for _ in range(4):
            cluster.record_health(7, healthy=False)

        status = cluster.get_cluster_status()

        self.assertEqual(status["open"], 1)
        self.assertEqual(len(status["isolation_events"]), 1)
        self.assertEqual(status["isolation_events"][0]["gpu_id"], 7)


if __name__ == "__main__":
    unittest.main()
