# `nosv`

`nosv` is a safe, futures-based Rust runtime layered over `nosv-sys` and
nOS-V. It supports ordinary `async fn` syntax without exposing native task
descriptors to safe code.

This crate remains experimental and has not been battle-tested in production.
It is nevertheless beyond an initial runtime prototype: the executor, timers,
`io_uring` driver, and owned-buffer file and TCP layers have automated coverage,
including focused cancellation and shutdown cases, through deterministic tests,
tests against a real nOS-V runtime and kernel ring, scheduled stress and
sanitizer jobs, and a self-verifying concurrent TCP example. That is evidence
for the scenarios described in [Verification](#verification) and
[`TESTING.md`](TESTING.md), not a production-readiness guarantee.

The current implementation includes:

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
  pointer-context `user_data`, typed CQE delivery, and explicit cancellation draining;
- safe `io`, `fs`, and numeric-address `net` layers with owned buffers, positional
  file I/O, TCP connect/accept/send/receive, and optional Tokio I/O traits;
- a runnable framed-TCP proof of concept with concurrent loopback clients,
  bounded frames, operation deadlines, and cooperative failure cleanup.

## Task example

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

## Network and I/O proof of concept

[`examples/net_io_poc.rs`](examples/net_io_poc.rs) is a self-contained example
of the safe `net` and `io` layers. It binds a numeric loopback address, launches
four clients, and gives each accepted connection its own nOS-V task. Messages use
a four-byte big-endian length prefix and owned `Vec<u8>` buffers; the server
uppercases each payload and every client verifies its response.

The example deliberately passes an `IoHandle` to `TcpListener::bind_on` and
`TcpStream::connect_on`, caps frames at 64 KiB before allocating, applies a
five-second deadline to each network operation, and aborts and drains outstanding
tasks on failure. It uses the default `time` and `io-uring` features and needs no
Tokio compatibility layer or additional dependency.

```text
cargo run --example net_io_poc
```

Its framing checks can be run without initializing a native runtime:

```text
cargo test --example net_io_poc
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

Default features are `rt`, `time`, and `io-uring`. The I/O feature creates one ring and
non-parallel nOS-V driver task for every runtime. `nosv::io_uring::IoUringConfig`
controls queue depth, CQ reap size, polling target, and per-submission buffering.
`Runtime<S, C>` selects the fork's sealed small or large entry markers, and
`IoUringHandle::submit_entries` provides an unsafe raw-SQE iterator API. The
runtime replaces each SQE's `user_data` with a stable context pointer and
restores the caller value on typed CQEs. `IoHandle`, `fs::File`, `net::TcpStream`,
and `net::TcpListener` provide the safe, non-generic layer across every supported
entry width. Owned-buffer operations retain resources through terminal completion;
drop requests cancellation, although a kernel side effect may still win that race.
Enable `tokio-compat` for Tokio `AsyncRead` and `AsyncWrite` on TCP streams.
`futures-io` and `native-sync` continue to reserve later ecosystem layers.

The current platform baseline is Linux, `std`, Rust 1.92+, and nOS-V 4.1+.
The crate is GPL-3.0-only to match the current `nosv-sys` package.

## Verification

The deterministic and native integration suites are documented in
[`TESTING.md`](TESTING.md). The tests exercise runtime init/reinit, topology and
memory queries, borrowed `block_on`, cross-thread spawn, wake storms and stale
wakers, panic capture, abort, timer ordering and cancellation, raw `io_uring`
admission and cancellation draining, owned-buffer file I/O, numeric TCP
fallback and loopback traffic, optional Tokio I/O traits, and shutdown. The
native cases use a real nOS-V process and kernel ring rather than a mock.

```text
PKG_CONFIG_PATH=/path/to/prefix/lib/pkgconfig \
LD_LIBRARY_PATH=/path/to/prefix/lib \
cargo test --all-features -- --test-threads=1
```

For a shorter end-to-end check of runtime tasks plus safe TCP and owned-buffer
I/O, run `timeout 30s cargo run --example net_io_poc` with the same build and
loader environment.
