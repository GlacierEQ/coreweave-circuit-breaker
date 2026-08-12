# CoreWeave Circuit Breaker

A GPU isolation and controlled-recovery component for preventing repeatedly failing devices from remaining in the schedulable pool.

## Purpose

The circuit breaker turns health observations into an explicit state machine:

```text
CLOSED --failure threshold--> OPEN --recovery timeout + acquired probe--> HALF_OPEN
HALF_OPEN --success--> CLOSED
HALF_OPEN --failure--> OPEN
```

The repository is independent GlacierEQ work. It does not claim CoreWeave production deployment or access to CoreWeave infrastructure.

## What is real

- configurable failure threshold and recovery timeout
- per-GPU circuit state
- pure read-only status queries
- explicit recovery-probe acquisition
- one half-open probe at a time
- timeout restart after failed probes
- cluster state counts for closed/open/half-open GPUs
- isolation and recovery-probe receipts
- JSON event runner
- installed `coreweave-circuit-breaker` CLI
- behavior tests covering failure and recovery paths

## Install and run

```bash
python -m pip install -e ".[dev]"
coreweave-circuit-breaker demo
pytest
```

Process a JSON event stream:

```bash
coreweave-circuit-breaker run \
  --events events.json \
  --failure-threshold 3 \
  --recovery-timeout 300
```

Event types are `health` and `probe`. Explicit event timestamps make simulation and recovery tests reproducible.

## Integration boundary

This repository decides whether a GPU should be isolated or admitted for a controlled probe. It does not drain a scheduler, quarantine hardware at the provider layer, page an operator, or move workloads. Those actions belong to an orchestrator consuming the receipt/state contract.
