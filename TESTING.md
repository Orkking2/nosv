# Testing

The suite is split between deterministic state tests and lifecycle-sensitive tests that execute a
real nOS-V runtime and kernel `io_uring` instance.

## Required local gates

Run these from the `nosv` checkout with `nosv-sys` and nOS-V in sibling directories. Set
`PKG_CONFIG_PATH` and the loader path as described in `README.md`.

```sh
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
cargo check --all-targets --no-default-features
cargo check --all-targets --no-default-features --features rt
cargo check --all-targets --no-default-features --features time
cargo check --all-targets --no-default-features --features io-uring
cargo check --all-targets --no-default-features --features futures-io
cargo check --all-targets --no-default-features --features native-sync
cargo check --all-targets --all-features
timeout 180s cargo test --all-features -- --test-threads=1
cargo test --example net_io_poc
timeout 30s cargo run --example net_io_poc
```

The native lane is intentionally serialized because nOS-V initialization and configuration are
process-wide. It must fail if the host cannot create an `io_uring`; capability failures are not
silently skipped.

## Coverage map

Pure deterministic tests cover native error translation, topology and affinity bounds, timer poll
ordering, every ring configuration field, stable pointer-context metadata, and driver park
decisions. Clippy, rustdoc, feature-matrix, and MSRV jobs compile the public generic API.

Real native tests cover fork rejection, initialization/reinitialization, borrowed roots, spawned
and cross-thread tasks, wake storms, stale wakers, panic and abort, and timer behavior. The I/O
lane additionally verifies caller `user_data` restoration from `block_on`, Entry128/Entry32 rings,
concurrent spawned NOP bursts beyond both ring depth and reap size, and AsyncCancel results plus
original-CQE draining for an owned long timeout. The proof-of-concept example adds pure
frame-boundary tests and a live, self-verifying run with four concurrent loopback clients, owned
buffers, numeric TCP, per-operation deadlines, and explicit task draining. Wall-clock time is used
only as an outer deadlock guard.

GitHub Actions pins both native dependencies and uses `ubuntu-24.04`. Pull requests run the full
required matrix and Rust 1.92 checks. A scheduled sanitized build repeats cancellation, wake, and
shutdown races under nOS-V ASan/UBSan instrumentation.

## Safe I/O layers

`tests/io_layers.rs` serializes native runtime tests for owned-buffer file I/O,
append validation, large entry-width erasure, numeric TCP fallback, loopback
send/receive, pending-accept cancellation, and the optional Tokio extension
traits. Run the Tokio-specific gate with:

```text
cargo test --features tokio-compat --test io_layers -- --test-threads=1
```

## Network and I/O proof of concept

`examples/net_io_poc.rs` is the runnable smoke application for the safe network
layer. `cargo test --example net_io_poc` checks the four-byte frame encoding and
the 64 KiB local and peer limits without starting nOS-V. The live command starts
the runtime and its `io_uring` and timer drivers, performs four concurrent framed
request/response exchanges, verifies every transformed payload, drains all tasks,
and shuts the runtime down:

```text
timeout 30s cargo run --example net_io_poc
```
