# `nosv`

`nosv` is a safe, futures-based Rust runtime layered over `nosv-sys` and
nOS-V. It supports ordinary `async fn` syntax without exposing native task
descriptors to safe code.

The implemented v0.1 foundation includes:

- owner-thread-checked runtime initialization and cooperative shutdown;
- `Send + 'static` future spawning from any thread;
- Rust-native `JoinHandle`, detachment, panic capture, and cooperative abort;
- a `Wake`-based scheduler with duplicate-wake coalescing;
- borrowed, non-`Send` root futures through attached-thread `block_on`;
- `yield_now`, current-runtime TLS, and scoped current CPU/NUMA queries;
- checked topology, affinity, and shared-memory wrappers;
- one non-parallel nOS-V timer task per runtime, providing `sleep`,
  `sleep_until`, and `timeout`;
- an optional runtime-wide `io_uring` driver with asynchronous SQ admission,
  pointer-context `user_data`, typed CQE delivery, and explicit cancellation draining.

## Example

```rust
use nosv::{Runtime, task};

async fn calculate(input: u64) -> u64 {
    task::yield_now().await;
    input * 2
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = Runtime::new()?;
    let handle = runtime.handle();

    runtime.block_on(async {
        let first = handle.spawn(calculate(20))?;
        let second = handle.spawn(calculate(11))?;
        assert_eq!(first.await? + second.await?, 62);
        Ok::<_, Box<dyn std::error::Error>>(())
    })?;

    runtime.shutdown()?;
    Ok(())
}
```

## Build and loader setup

`nosv-sys` discovers nOS-V using `pkg-config`. For a local source installation:

```text
PKG_CONFIG_PATH=/path/to/nos-v/prefix/lib/pkgconfig cargo build
LD_LIBRARY_PATH=/path/to/nos-v/prefix/lib ./target/debug/your-program
```

The nOS-V support/interposition library may also impose link-order or
`LD_PRELOAD` requirements. Follow the nOS-V installation's README for the
chosen configuration; a dependency crate's `.cargo/config.toml` does not set
environment variables for this top-level package.

## Safety and scheduling contract

- `Runtime` is neither `Send` nor `Sync`; nOS-V init/shutdown pairing stays on
  the creating thread. `Handle` is `Send + Sync`.
- A spawned future is `Send + 'static` because nOS-V may resume its task on a
  different pthread. `block_on` polls its root future on the attached caller,
  so that root may borrow local non-`Send` state.
- Scheduling and abort are cooperative. A future that does not return from
  `poll` cannot be preempted and can make shutdown wait indefinitely.
- Dropping a `JoinHandle` detaches; call `abort` explicitly to request cancel.
- The native descriptor gate is held across every `nosv_submit` and
  `nosv_destroy`, preventing late wakers from racing descriptor retirement.
- Join completion is published only after the future is dropped and its native
  descriptor is destroyed. No unwind crosses a C callback.
- Handles inherited across `fork` make no native calls and report an error.
- Native configuration is process-global. Some nOS-V configurations enable
  FTZ/DAZ behavior, which changes strict IEEE-754 floating-point semantics.

Native stackful synchronization primitives are intentionally not exposed from
async task context: an unrelated Rust wake can conflict with their submit-based
unblock protocol. Async synchronization should be implemented by queuing Rust
wakers instead.

## Feature and release scope

Default features are `rt` and `time`. Enabling `io-uring` creates one ring and
non-parallel nOS-V driver task for every runtime. `nosv::io_uring::IoUringConfig`
controls queue depth, CQ reap size, polling target, and per-submission buffering.
`Runtime<S, C>` selects the fork's sealed small or large entry markers, and
`IoUringHandle::submit_entries` provides an unsafe raw-SQE iterator API. The
runtime replaces each SQE's `user_data` with a stable context pointer and
restores the caller value on typed CQEs. Safe `fs`, `net`, `io`, and owned-buffer
operations remain future layers.
`futures-io` and `native-sync` continue to reserve later ecosystem layers.

The initial platform baseline is Linux, `std`, Rust 1.85+, and nOS-V 4.1+.
The crate is GPL-3.0-only to match the current `nosv-sys` package.

## Verification

The deterministic and integration suites are documented in [`TESTING.md`](TESTING.md).
The integration tests run lifecycle-sensitive behavior in a real nOS-V process: init/reinit, topology and memory queries, borrowed `block_on`,
cross-thread spawn, ten thousand self-wakes, panic capture, abort, deadline
interruption, equal-deadline timers, timeout, timer cancellation, and shutdown.

```text
PKG_CONFIG_PATH=/path/to/prefix/lib/pkgconfig \
LD_LIBRARY_PATH=/path/to/prefix/lib \
cargo test --all-features -- --test-threads=1
```
