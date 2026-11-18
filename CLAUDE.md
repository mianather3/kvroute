# kvroute — Engineering Standards

These rules apply to every phase of work in this repository.

## Code style
- Idiomatic, clean systems Rust. No enterprise abstractions, trait proliferation, or macro bloat.
- Add a trait only when there are two or more real implementations today.
- Concise variable names. Comments are high-signal only: explain *why*, invariants, and non-obvious constraints, never restate the code.

## Crate boundaries
- `kvroute-core` stays pure, synchronous, and deterministic. No `tokio`, no async, no I/O, no wall-clock time, no global RNG. Randomness and time are passed in by the caller.
- `kvroute-sim` and `kvroute-proxy` must share identical routing logic through `kvroute-core`. Routing decisions are never reimplemented in either binary crate.

## Proxy hot path
- Lock-free reads via `arc-swap`; a single writer publishes state updates.
- No `Mutex`/`RwLock` contention on request routing.

## Gate (every phase)
All of the following must pass before a phase is done:

```sh
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

`make check && make test` runs the same gate.
