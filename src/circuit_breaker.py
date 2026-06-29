"""CoreWeave Circuit Breaker — Prevents cascading failures across GPUs.

Their pain: Cluster reliability at scale — one GPU failure cascades.

Innovation: 3-strike zone isolation with countdown recovery.
Isolates failing GPUs before they crash the whole cluster.
"""

import time
import json
from dataclasses import dataclass
from typing import Dict


class CircuitState:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class GPUCircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 300.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = 0.0

    def record_success(self):
        self.failure_count = max(0, self.failure_count - 1)
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True

    def get_status(self) -> dict:
        return {"state": self.state, "failures": self.failure_count, "can_execute": self.can_execute()}


class ClusterCircuitBreaker:
    def __init__(self):
        self.gpu_breakers: Dict[int, GPUCircuitBreaker] = {}
        self._isolation_log: list = []

    def evaluate_gpu(self, gpu_id: int, healthy: bool) -> dict:
        if gpu_id not in self.gpu_breakers:
            self.gpu_breakers[gpu_id] = GPUCircuitBreaker()

        breaker = self.gpu_breakers[gpu_id]
        if healthy:
            breaker.record_success()
        else:
            breaker.record_failure()

        if not breaker.can_execute():
            self._isolation_log.append({"gpu_id": gpu_id, "timestamp": time.time()})

        return {"gpu_id": gpu_id, "state": breaker.state, "can_execute": breaker.can_execute()}

    def get_cluster_status(self) -> dict:
        total = len(self.gpu_breakers)
        healthy = sum(1 for b in self.gpu_breakers.values() if b.state == CircuitState.CLOSED)
        return {"total_gpus": total, "healthy": healthy, "isolated": total - healthy}


if __name__ == "__main__":
    cb = ClusterCircuitBreaker()
    for i in range(8):
        result = cb.evaluate_gpu(i, i != 3)
        print(f"GPU {i}: {result['state']}")
    print(json.dumps(cb.get_cluster_status(), indent=2))
