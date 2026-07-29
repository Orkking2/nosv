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
original-CQE draining for an owned long timeout. Wall-clock time is used only as an outer deadlock
guard.

GitHub Actions pins both native dependencies and uses `ubuntu-24.04`. Pull requests run the full
required matrix and Rust 1.85 checks. A scheduled sanitized build repeats cancellation, wake, and
shutdown races under nOS-V ASan/UBSan instrumentation.
