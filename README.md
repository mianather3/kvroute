# kvroute

KV-cache-aware request routing for LLM inference fleets. kvroute sends each request to the backend most likely to already hold its prompt prefix in KV cache, while keeping backends within their KV memory budget.

## Architecture

| Crate | Role |
|-------|------|
| `kvroute-core` | Pure, synchronous, deterministic routing logic. No async runtime. |
| `kvroute-sim` | Offline simulator that replays traces through `kvroute-core` to evaluate policies. |
| `kvroute-proxy` | Tokio/axum proxy that applies the same `kvroute-core` logic to live traffic. |

The simulator and proxy share one routing implementation, so simulated results carry over to production behaviour.

## Core mechanisms

- **Chained 512-token block hashing.** Prompts are split into 512-token blocks; each block's hash folds in the previous block's hash, so a block hash identifies the whole prefix up to that point. Prefix matching becomes a walk over hashes.
- **Staleness and eviction fault modeling.** The router's view of each backend's cache can lag or be wrong because of delayed updates and evictions it hasn't seen. These faults are modeled explicitly so routing holds up under imperfect information.
- **KV-budget admission control.** Requests are admitted only when the target backend has KV memory headroom, so cache affinity can't overload a single node.

## Build and test

```sh
make check   # cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings
make test    # cargo test --workspace
make build   # cargo build --workspace --release
```

## License

MIT
