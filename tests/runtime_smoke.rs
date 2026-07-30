use nosv::{Runtime, task};
#[cfg(feature = "time")]
use std::time::Duration;
use std::{
    cell::Cell,
    future::Future,
    pin::Pin,
    rc::Rc,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll},
};

/// Self-waking future used to stress repeated suspend/resubmit epochs.
struct WakeMany(
    /// Number of pending/self-wake cycles remaining.
    u32,
);

impl Future for WakeMany {
    /// The stress future produces no value.
    type Output = ();

    /// Consumes one cycle, wakes itself, and becomes ready when all cycles finish.
    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
        if self.0 == 0 {
            Poll::Ready(())
        } else {
            self.0 -= 1;
            context.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

/// Future whose captured waker is exercised concurrently by foreign pthreads.
struct ExternalWake {
    /// Release flag written after the wake storm has completed.
    ready: Arc<AtomicBool>,
    /// Slot through which the test extracts the executor-created waker.
    slot: Arc<Mutex<Option<std::task::Waker>>>,
}

impl Future for ExternalWake {
    /// The cross-thread wake check produces no value.
    type Output = ();

    /// Completes after release or refreshes the externally accessible waker slot.
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
        if self.ready.load(Ordering::Acquire) {
            Poll::Ready(())
        } else {
            *self.slot.lock().unwrap() = Some(context.waker().clone());
            Poll::Pending
        }
    }
}

/// Ready future that deliberately leaves a cloned waker alive after completion.
struct ReadyWithWaker(
    /// Storage used to fire the stale waker after native descriptor retirement.
    Arc<Mutex<Option<std::task::Waker>>>,
);

impl Future for ReadyWithWaker {
    /// The stale-waker setup produces no value.
    type Output = ();

    /// Captures the current waker and completes in the same poll epoch.
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
        *self.0.lock().unwrap() = Some(context.waker().clone());
        Poll::Ready(())
    }
}

/// Output value that records when a detached task's result is destroyed.
struct DropFlag(
    /// Flag set by the destructor with release ordering.
    Arc<AtomicBool>,
);

impl Drop for DropFlag {
    /// Publishes proof that detached output cleanup ran before runtime shutdown returned.
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

/// Exercises the current public runtime lifecycle against a real nOS-V installation.
///
/// The scenario covers fork guards, topology and memory queries, a borrowed root future, task
/// attributes, self/foreign/stale wakes, detached outputs, panic capture, cooperative abort, timer
/// ordering and cancellation, shutdown draining, closed capabilities, and full reinitialization.
#[test]
fn runtime_spawn_join_abort_and_reinitialize() {
    let runtime = Runtime::new().expect("initialize nOS-V");
    let handle = runtime.handle();

    // SAFETY: the child performs only the guarded handle operation and _exit;
    // the parent synchronously reaps it before continuing the test.
    unsafe {
        let child = libc::fork();
        assert!(child >= 0);
        if child == 0 {
            let guarded = matches!(handle.spawn(async {}), Err(nosv::SpawnError::ForkedProcess));
            libc::_exit(i32::from(!guarded));
        }
        let mut status = 0;
        assert_eq!(libc::waitpid(child, &mut status, 0), child);
        assert!(libc::WIFEXITED(status));
        assert_eq!(libc::WEXITSTATUS(status), 0);
    }

    let topology = handle.topology().unwrap();
    let cpus = topology.cpus().unwrap();
    assert!(!cpus.is_empty());
    let preferred_cpu = cpus[0];
    let memory = handle.memory_stats().unwrap();
    assert!(memory.used <= memory.size);

    let borrowed = Rc::new(Cell::new(0_u32));
    let borrowed_root = borrowed.clone();
    let output_dropped = Arc::new(AtomicBool::new(false));
    runtime.block_on(async {
        borrowed_root.set(4);

        let first = handle
            .spawn(async {
                task::yield_now().await;
                40_u32
            })
            .unwrap();
        let second = handle
            .task()
            .rust_name("answer tail")
            .priority(1)
            .affinity(nosv::Affinity::preferred_cpu(preferred_cpu))
            .monitoring_cost(17)
            .spawn(async { 2_u32 })
            .unwrap();
        assert_eq!(first.await.unwrap() + second.await.unwrap(), 42);

        handle.spawn(WakeMany(10_000)).unwrap().await.unwrap();

        let ready = Arc::new(AtomicBool::new(false));
        let slot = Arc::new(Mutex::new(None));
        let externally_woken = handle
            .spawn(ExternalWake {
                ready: ready.clone(),
                slot: slot.clone(),
            })
            .unwrap();
        while slot.lock().unwrap().is_none() {
            task::yield_now().await;
        }
        let waker = slot.lock().unwrap().as_ref().unwrap().clone();
        let mut wake_threads = Vec::new();
        for _ in 0..8 {
            let waker = waker.clone();
            wake_threads.push(std::thread::spawn(move || {
                for _ in 0..1_000 {
                    waker.wake_by_ref();
                }
            }));
        }
        for thread in wake_threads {
            thread.join().unwrap();
        }
        ready.store(true, Ordering::Release);
        waker.wake();
        externally_woken.await.unwrap();

        let stale_slot = Arc::new(Mutex::new(None));
        handle
            .spawn(ReadyWithWaker(stale_slot.clone()))
            .unwrap()
            .await
            .unwrap();
        stale_slot.lock().unwrap().take().unwrap().wake();

        let detached_output = handle
            .spawn({
                let output_dropped = output_dropped.clone();
                async move { DropFlag(output_dropped) }
            })
            .unwrap();
        drop(detached_output);

        let external = {
            let handle = handle.clone();
            std::thread::spawn(move || handle.spawn(async { 9_u32 }).unwrap())
                .join()
                .unwrap()
        };
        assert_eq!(external.await.unwrap(), 9);

        let panic_task = handle
            .spawn(async { panic!("captured task panic") })
            .unwrap();
        assert!(panic_task.await.unwrap_err().is_panic());

        let cancelled = handle.spawn(std::future::pending::<()>()).unwrap();
        assert!(cancelled.abort());
        assert!(cancelled.await.unwrap_err().is_cancelled());

        #[cfg(feature = "time")]
        {
            nosv::time::timeout(
                Duration::from_secs(1),
                nosv::time::sleep(Duration::from_millis(2)),
            )
            .await
            .unwrap();

            let late = handle
                .spawn(async {
                    nosv::time::sleep(Duration::from_millis(20)).await;
                    20_u32
                })
                .unwrap();
            task::yield_now().await;
            let early = handle
                .spawn(async {
                    nosv::time::sleep(Duration::from_millis(1)).await;
                    1_u32
                })
                .unwrap();
            assert_eq!(early.await.unwrap(), 1);
            assert_eq!(late.await.unwrap(), 20);

            let mut equal = Vec::new();
            for _ in 0..128 {
                equal.push(
                    handle
                        .spawn(async { nosv::time::sleep(Duration::from_millis(2)).await })
                        .unwrap(),
                );
            }
            for timer in equal {
                timer.await.unwrap();
            }

            assert!(
                nosv::time::timeout(Duration::from_millis(2), std::future::pending::<()>())
                    .await
                    .is_err()
            );
            let long = handle
                .spawn(async { nosv::time::sleep(Duration::from_secs(30)).await })
                .unwrap();
            assert!(long.abort());
            assert!(long.await.unwrap_err().is_cancelled());
        }
    });
    assert_eq!(borrowed.get(), 4);
    let detached_pending = handle.spawn(std::future::pending::<()>()).unwrap();
    drop(detached_pending);
    runtime.shutdown().unwrap();
    assert!(output_dropped.load(Ordering::Acquire));
    assert_eq!(topology.cpus(), Err(nosv::NativeError::NotInitialized));

    let runtime = Runtime::new().expect("reinitialize nOS-V");
    assert_eq!(runtime.block_on(async { 3_u32 }), 3);
    runtime.shutdown().unwrap();
}
