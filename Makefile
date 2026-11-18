.PHONY: check test build bench

check:
	cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings

test:
	cargo test --workspace

build:
	cargo build --workspace --release

bench:
	@echo "bench: not yet implemented"
