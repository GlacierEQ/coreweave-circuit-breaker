# CoreWeave Circuit Breaker

A private GlacierEQ R&D prototype for **explicit per-GPU circuit-breaker state and timeout-gated recovery probes**.

## Implemented mechanism

The breaker now separates status reads from state transitions:

- repeated failures open a GPU breaker at a configurable threshold;
- an open breaker remains non-executable until its recovery timeout has elapsed;
- timeout readiness is observable without changing state;
- moving to half-open requires an explicit recovery-probe transition;
- a successful half-open probe closes and resets the breaker;
- a failed half-open probe reopens it;
- cluster isolation events are logged once when a breaker crosses into open state;
- the clock is injectable so transition behavior can be tested deterministically.

## Truth boundary

This repository is not evidence of CoreWeave affiliation, production deployment, prevention of cascading GPU failures, or improved cluster availability. It implements a bounded circuit-breaker primitive whose production value still requires integration and fault-injection evidence.

The previous example also failed to reach its own default three-failure threshold. The current example deliberately demonstrates open → half-open → closed recovery.

## Verification

```bash
python -m unittest discover -s tests -v
python src/circuit_breaker.py
```

CI targets the actual default branch, `master`.

## Promotion gates

Before canonical integration:

- positive hosted tests;
- concurrency/thread-safety decisions;
- explicit policy for rolling failure windows versus lifetime failure count;
- fault-injection scenarios across multiple GPUs;
- integration with a scheduler or workload-drain action;
- measured comparison against an ungated retry baseline.
