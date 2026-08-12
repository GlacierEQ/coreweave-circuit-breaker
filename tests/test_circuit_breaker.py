from __future__ import annotations

import json
import subprocess
import sys

from circuit_breaker import CircuitState, ClusterCircuitBreaker, GPUCircuitBreaker, demo


def test_three_failures_open_and_status_is_read_only():
    breaker = GPUCircuitBreaker(failure_threshold=3, recovery_timeout=10)
    breaker.record_failure(0)
    breaker.record_failure(1)
    breaker.record_failure(2)

    before = breaker.state
    first = breaker.status(20)
    second = breaker.status(20)

    assert before is CircuitState.OPEN
    assert first == second
    assert breaker.state is CircuitState.OPEN
    assert first["recovery_ready"] is True
    assert first["can_execute"] is True


def test_only_acquire_transitions_open_breaker_to_half_open():
    breaker = GPUCircuitBreaker(failure_threshold=1, recovery_timeout=10)
    breaker.record_failure(2)

    assert breaker.acquire(11) is False
    assert breaker.state is CircuitState.OPEN
    assert breaker.acquire(12) is True
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.acquire(12) is False


def test_successful_probe_closes_circuit():
    breaker = GPUCircuitBreaker(failure_threshold=1, recovery_timeout=0)
    breaker.record_failure(1)
    assert breaker.acquire(1) is True
    breaker.record_success(1)

    assert breaker.state is CircuitState.CLOSED
    assert breaker.failure_count == 0
    assert breaker.can_execute(1) is True


def test_failed_probe_reopens_and_restarts_timeout():
    breaker = GPUCircuitBreaker(failure_threshold=1, recovery_timeout=10)
    breaker.record_failure(0)
    assert breaker.acquire(10) is True
    breaker.record_failure(10)

    assert breaker.state is CircuitState.OPEN
    assert breaker.recovery_ready(19.9) is False
    assert breaker.recovery_ready(20) is True


def test_cluster_logs_one_isolation_transition_and_recovery_probe():
    cluster = ClusterCircuitBreaker(failure_threshold=2, recovery_timeout=5)
    cluster.record_health("gpu-1", False, 0)
    cluster.record_health("gpu-1", False, 1)
    cluster.record_health("gpu-1", False, 2)
    assert cluster.acquire_gpu("gpu-1", 6) is True
    cluster.record_health("gpu-1", True, 6)

    receipt = cluster.receipt(6)
    assert len(receipt["isolation_log"]) == 1
    assert len(receipt["recovery_log"]) == 1
    assert receipt["cluster"]["states"]["closed"] == 1


def test_cluster_summary_distinguishes_open_and_half_open():
    cluster = ClusterCircuitBreaker(failure_threshold=1, recovery_timeout=0)
    cluster.record_health("open", False, 0)
    cluster.record_health("half", False, 0)
    assert cluster.acquire_gpu("half", 0) is True

    status = cluster.get_cluster_status(0)
    assert status["states"] == {"closed": 0, "open": 1, "half_open": 1}
    assert status["schedulable"] == 1


def test_demo_exercises_isolation_and_recovery():
    payload = demo()
    assert payload["cluster"]["isolations"] == 1
    assert payload["cluster"]["recovery_probes"] == 1
    assert payload["gpus"]["gpu-3"]["state"] == "closed"


def test_module_cli_demo_executes():
    completed = subprocess.run(
        [sys.executable, "-m", "circuit_breaker", "demo"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["cluster"]["total_gpus"] == 2
