.PHONY: run
run:
	RUST_LOG=info cargo run

.PHONY: fmt
fmt:
	cargo fmt