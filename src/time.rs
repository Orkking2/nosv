//! Cancellation-safe monotonic timers driven by one nOS-V task per runtime.

use crate::{ffi, runtime::Handle, util::lock};
use std::{
    cmp::Ordering,
    collections::BinaryHeap,
    error::Error,
    fmt,
    future::Future,
    panic::{self, AssertUnwindSafe},
    pin::Pin,
    ptr,
    sync::{
        Arc, Condvar, Mutex, MutexGuard, PoisonError,
        atomic::{AtomicU64, Ordering as AtomicOrdering},
    },
    task::{Context, Poll, Waker},
    time::{Duration, Instant},
};

/// State shared by one [`Sleep`] future and the timer driver.
///
/// Firing, cancellation, and waker replacement use a single mutex so the driver and future cannot
/// lose a wake or publish a timer after its future has been dropped.
struct TimerState {
    /// Whether the deadline has been observed and completion may be returned.
    fired: bool,
    /// Whether dropping the future made this heap entry stale.
    cancelled: bool,
    /// Most recent async task waiting for this timer.
    waiter: Option<Waker>,
}

/// Heap record connecting a monotonic deadline to shared timer state.
struct TimerEntry {
    /// Monotonic instant at which the timer becomes ready.
    deadline: Instant,
    /// Tie-breaker giving otherwise equal deadlines a total ordering.
    id: u64,
    /// State retained until the driver removes this record from its heap.
    state: Arc<Mutex<TimerState>>,
}

impl PartialEq for TimerEntry {
    /// Compares the deadline and unique tie-breaker used by the heap ordering.
    fn eq(&self, other: &Self) -> bool {
        self.deadline == other.deadline && self.id == other.id
    }
}
impl Eq for TimerEntry {}
impl PartialOrd for TimerEntry {
    /// Delegates to the total ordering because every [`Instant`] and timer ID is comparable.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TimerEntry {
    /// Reverses chronological order so [`BinaryHeap`] behaves as a minimum-deadline heap.
    ///
    /// IDs are reversed as well to preserve a stable total order for equal deadlines.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .deadline
            .cmp(&self.deadline)
            .then_with(|| other.id.cmp(&self.id))
    }
}

/// Timer collection and shutdown request consumed by the stackful driver loop.
struct DriverData {
    /// Minimum-deadline heap; cancelled entries are removed lazily.
    timers: BinaryHeap<TimerEntry>,
    /// Whether the loop must cancel outstanding timers and return.
    shutdown: bool,
}

/// Native parking state used to select the correct early-wake submission mode.
///
/// nOS-V distinguishes waking an indefinite pause from interrupting `nosv_waitfor`. Tracking that
/// distinction under one gate lets registration race either operation without losing a wake.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DriverPark {
    /// Rust ownership exists, but descriptor initialization is incomplete.
    Building,
    /// The callback is inspecting timers or invoking wakers.
    Running,
    /// The callback is indefinitely parked with no deadline.
    Paused,
    /// The callback is parked in a relative deadline wait.
    DeadlineWait,
    /// A submit has been issued to interrupt the current park operation.
    WakePending,
    /// Shutdown made the run callback return without parking again.
    Terminal,
    /// The completed callback destroyed the descriptor.
    Destroyed,
}

/// Descriptor state serialized across driver parking, notification, and retirement.
struct DriverGate {
    /// Native timer-task descriptor, removed immediately before destruction.
    raw: Option<ffi::RawTask>,
    /// Current callback or park phase.
    phase: DriverPark,
    /// Whether new work arrived and must be observed before the next park.
    notified: bool,
}

/// One stackful native task that services every timer belonging to a runtime.
///
/// The driver uses nOS-V's `pause` and `waitfor` primitives only inside this dedicated synchronous
/// callback. Ordinary async futures never enter stackful blocking primitives, avoiding interference
/// between unrelated Rust wakes and native unblock operations.
pub(crate) struct TimerDriver {
    /// Deadline heap and shutdown request.
    data: Mutex<DriverData>,
    /// Native descriptor and early-wake protocol.
    gate: Mutex<DriverGate>,
    /// Monotonic tie-breaker allocated to timer entries.
    next_id: AtomicU64,
    /// Whether the completed callback has retired the native descriptor.
    completed: Mutex<bool>,
    /// Notifies owner-thread shutdown after descriptor retirement.
    completed_cv: Condvar,
}

impl TimerDriver {
    /// Creates and initially submits the runtime's dedicated timer task.
    ///
    /// A boxed `Arc<TimerDriver>` is stored as one potentially unaligned metadata pointer. Initial
    /// submission failure reclaims both the descriptor and metadata owner because the completion
    /// callback cannot yet run.
    pub(crate) fn start(task_type: ffi::RawTaskType) -> Result<Arc<Self>, crate::NativeError> {
        let driver = Arc::new(Self {
            data: Mutex::new(DriverData {
                timers: BinaryHeap::new(),
                shutdown: false,
            }),
            gate: Mutex::new(DriverGate {
                raw: None,
                phase: DriverPark::Building,
                notified: false,
            }),
            next_id: AtomicU64::new(1),
            completed: Mutex::new(false),
            completed_cv: Condvar::new(),
        });
        let raw = ffi::create(task_type, std::mem::size_of::<*const Self>())?;
        let metadata = match ffi::metadata(raw) {
            Ok(metadata) => metadata,
            Err(error) => {
                let _ = ffi::destroy(raw);
                return Err(error);
            }
        };
        let owner = Arc::into_raw(driver.clone());
        // SAFETY: the metadata is pointer-sized but may be unaligned. The
        // completed callback is the sole consumer of this pointer.
        unsafe { ptr::write_unaligned(metadata.as_ptr().cast::<*const Self>(), owner) };

        {
            let mut gate = driver.lock_gate();
            gate.raw = Some(raw);
            gate.phase = DriverPark::Running;
        }

        if let Err(error) = {
            let gate = driver.lock_gate();
            ffi::submit(gate.raw.expect("timer descriptor initialized"))
        } {
            let mut gate = driver.lock_gate();
            let raw = gate.raw.take().expect("timer descriptor initialized");
            let _ = ffi::destroy(raw);
            gate.phase = DriverPark::Destroyed;
            // SAFETY: submission failed, so completion cannot consume the owner.
            unsafe { drop(Arc::from_raw(owner)) };
            return Err(error);
        }

        Ok(driver)
    }

    /// Locks the timer heap, recovering state if an internal mutex was poisoned.
    fn lock_data(&self) -> MutexGuard<'_, DriverData> {
        lock(&self.data)
    }

    /// Locks the native park gate, recovering state if an internal mutex was poisoned.
    fn lock_gate(&self) -> MutexGuard<'_, DriverGate> {
        lock(&self.gate)
    }

    /// Adds a timer entry and wakes the driver so it can reconsider its next deadline.
    ///
    /// Registration after shutdown marks the future cancelled instead of retaining new work. IDs
    /// are never allowed to wrap because they participate in the heap's total ordering.
    fn register(&self, deadline: Instant, state: Arc<Mutex<TimerState>>) {
        let id = self.next_id.fetch_add(1, AtomicOrdering::Relaxed);
        if id == u64::MAX {
            invariant_abort("timer identifier exhausted");
        }

        {
            let mut data = self.lock_data();
            if data.shutdown {
                let mut state = lock(&state);
                state.cancelled = true;
                state.waiter = None;
                return;
            }
            data.timers.push(TimerEntry {
                deadline,
                id,
                state,
            });
        }

        self.notify();
    }

    /// Records pending driver work and interrupts a native park when necessary.
    ///
    /// An indefinite pause uses ordinary submit, whereas `DeadlineWait` requires deadline-wake
    /// submit. Holding the gate through submission prevents retirement from destroying `raw` while
    /// nOS-V is consuming it; duplicate notifications are coalesced by `notified`/`WakePending`.
    fn notify(&self) {
        let mut gate = self.lock_gate();
        gate.notified = true;

        let submit = match gate.phase {
            DriverPark::Paused => Some(false),
            DriverPark::DeadlineWait => Some(true),
            DriverPark::Running | DriverPark::WakePending => None,
            DriverPark::Building | DriverPark::Terminal | DriverPark::Destroyed => return,
        };

        if let Some(deadline) = submit {
            gate.phase = DriverPark::WakePending;
            let raw = gate
                .raw
                .unwrap_or_else(|| invariant_abort("timer descriptor missing"));
            let result = if deadline {
                ffi::submit_deadline_wake(raw)
            } else {
                ffi::submit(raw)
            };
            if let Err(error) = result {
                eprintln!("nOS-V timer wake failed: {error}");
                invariant_abort("timer wake submit failed");
            }
        }
    }

    /// Runs the stackful timer loop until shutdown is requested.
    ///
    /// Each pass removes expired and stale entries, invokes captured wakers outside all timer
    /// locks, and then atomically chooses between continuing, pausing indefinitely, or waiting for
    /// the next relative deadline. A shutdown pass cancels all retained timer states and returns so
    /// nOS-V can invoke the completed callback.
    fn run(&self) {
        loop {
            let (wait, wake) = {
                let mut data = self.lock_data();
                if data.shutdown {
                    for entry in data.timers.drain() {
                        let mut state = lock(&entry.state);
                        state.cancelled = true;
                        state.waiter = None;
                    }
                    (None, Vec::new())
                } else {
                    let now = Instant::now();
                    let mut wake = Vec::new();
                    while data
                        .timers
                        .peek()
                        .is_some_and(|entry| entry.deadline <= now)
                    {
                        let entry = data.timers.pop().expect("peeked timer");
                        let mut state = lock(&entry.state);
                        if !state.cancelled {
                            state.fired = true;
                            if let Some(waiter) = state.waiter.take() {
                                wake.push(waiter);
                            }
                        }
                    }
                    while data.timers.peek().is_some_and(|entry| {
                        entry
                            .state
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .cancelled
                    }) {
                        data.timers.pop();
                    }
                    let wait = data
                        .timers
                        .peek()
                        .map(|entry| entry.deadline.saturating_duration_since(now));
                    (wait, wake)
                }
            };

            for waiter in wake {
                let _ = panic::catch_unwind(AssertUnwindSafe(|| waiter.wake()));
            }
            if self.lock_data().shutdown {
                self.lock_gate().phase = DriverPark::Terminal;
                return;
            }

            let should_wait = {
                let mut gate = self.lock_gate();
                if gate.notified {
                    gate.notified = false;
                    gate.phase = DriverPark::Running;
                    false
                } else {
                    gate.phase = if wait.is_some() {
                        DriverPark::DeadlineWait
                    } else {
                        DriverPark::Paused
                    };
                    true
                }
            };
            if !should_wait {
                continue;
            }
            let result = match wait {
                Some(duration) => ffi::waitfor(duration),
                None => ffi::pause(),
            };
            if let Err(error) = result {
                eprintln!("nOS-V timer wait failed: {error}");
                invariant_abort("timer driver could not park");
            }
            self.lock_gate().phase = DriverPark::Running;
        }
    }

    /// Requests driver shutdown and waits until its native descriptor has been destroyed.
    ///
    /// Runtime shutdown calls this on the native initialization thread after async tasks have been
    /// asked to cancel. The wait is intentionally unbounded because calling native shutdown while
    /// a driver descriptor remains live would violate nOS-V ownership requirements.
    pub(crate) fn shutdown_and_wait(&self) {
        self.lock_data().shutdown = true;
        self.notify();
        let mut completed = self
            .completed
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        while !*completed {
            completed = self
                .completed_cv
                .wait(completed)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Destroys the terminal driver descriptor and releases shutdown waiters.
    ///
    /// Destruction occurs under the park gate, serializing it with every notification submit. The
    /// completion condition is set only after the descriptor is no longer usable.
    fn retire(&self) {
        {
            let mut gate = self.lock_gate();
            if gate.phase != DriverPark::Terminal {
                invariant_abort("timer completed before terminal state");
            }
            let raw = gate
                .raw
                .take()
                .unwrap_or_else(|| invariant_abort("timer descriptor missing"));
            if let Err(error) = ffi::destroy(raw) {
                eprintln!("nOS-V timer destroy failed: {error}");
                invariant_abort("timer descriptor could not be destroyed");
            }
            gate.phase = DriverPark::Destroyed;
        }

        *self
            .completed
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = true;
        self.completed_cv.notify_all();
    }
}

/// Reads the unaligned timer-owner pointer stored in descriptor metadata.
///
/// # Safety
///
/// `raw` must be the timer descriptor initialized by [`TimerDriver::start`], and its completed
/// callback must not yet have consumed the boxed `Arc`. Only that callback—or the proven initial
/// submission failure path—may reconstruct the pointer as a `Box`.
unsafe fn timer_owner(raw: ffi::RawTask) -> *const TimerDriver {
    let metadata = ffi::metadata(raw).unwrap_or_else(|_| invariant_abort("timer metadata missing"));
    // SAFETY: start wrote this Box pointer with write_unaligned.
    unsafe { ptr::read_unaligned(metadata.as_ptr().cast::<*const TimerDriver>()) }
}

/// nOS-V callback that enters the stackful timer service loop.
///
/// The whole body is panic-contained so Rust never unwinds across the C ABI. An unexpected internal
/// panic represents an unprovable descriptor state and therefore aborts the process.
///
/// # Safety
///
/// nOS-V must pass the live timer descriptor whose metadata contains the owner installed by
/// [`TimerDriver::start`]. The task type must be non-parallel.
pub(crate) unsafe extern "C" fn run_callback(pointer: nosv_sys::nosv_task_t) {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: nOS-V supplies the live timer descriptor.
        let raw = unsafe { ffi::RawTask::from_ptr(pointer) }
            .unwrap_or_else(|| invariant_abort("null timer task"));
        // SAFETY: owner remains allocated until completed callback.
        let driver = unsafe { timer_owner(raw).as_ref() }
            .unwrap_or_else(|| invariant_abort("null timer owner"));
        driver.run();
    }));
    if result.is_err() {
        invariant_abort("panic in timer run callback");
    }
}

/// nOS-V callback that consumes timer metadata and retires the driver descriptor.
///
/// Completion is signalled only after [`TimerDriver::retire`] destroys the descriptor, allowing
/// runtime shutdown to proceed without racing a late timer notification.
///
/// # Safety
///
/// nOS-V must invoke this exactly once after the timer run callback returns terminally, while the
/// initialized descriptor metadata remains readable.
pub(crate) unsafe extern "C" fn completed_callback(pointer: nosv_sys::nosv_task_t) {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: nOS-V supplies the terminal timer descriptor.
        let raw = unsafe { ffi::RawTask::from_ptr(pointer) }
            .unwrap_or_else(|| invariant_abort("null completed timer"));
        // SAFETY: this callback is the sole consumer of the Box pointer.
        let owner = unsafe { Arc::from_raw(timer_owner(raw)) };
        owner.retire();
    }));
    if result.is_err() {
        invariant_abort("panic in timer completed callback");
    }
}

/// Reports an impossible timer-driver transition and terminates the process.
///
/// Once descriptor ownership or park mode is uncertain, continuing could select the wrong native
/// wake mode or use freed storage. Aborting is the conservative memory-safe response.
fn invariant_abort(message: &str) -> ! {
    eprintln!("nOS-V timer invariant failed: {message}");
    std::process::abort()
}

/// A cancellation-safe monotonic deadline future.
///
/// Registration is lazy: constructing a `Sleep` does not require a current runtime, but its first
/// pending poll does. Dropping a registered sleep marks its heap entry stale; the driver retains
/// shared state until that entry is lazily removed, avoiding dangling pointers.
pub struct Sleep {
    /// Monotonic instant at which this future becomes ready.
    deadline: Instant,
    /// Fired/cancelled state shared with the driver heap entry.
    state: Arc<Mutex<TimerState>>,
    /// Driver chosen on first poll; retaining it keeps registered state valid.
    driver: Option<Arc<TimerDriver>>,
}

impl Sleep {
    /// Returns this timer's monotonic deadline.
    ///
    /// The value uses [`Instant`], so wall-clock changes do not alter the timer.
    pub const fn deadline(&self) -> Instant {
        self.deadline
    }
}

impl Future for Sleep {
    /// Timer completion carries no additional value.
    type Output = ();

    /// Checks readiness, refreshes the waiter, and lazily registers the deadline.
    ///
    /// Already-expired timers complete without consulting a runtime. A future that must wait is
    /// bound to the current runtime's driver on its first poll and remains with that driver.
    ///
    /// # Panics
    ///
    /// Panics if a non-expired sleep is first polled outside a running nOS-V runtime. This mirrors
    /// reactor-bound timer APIs: construction is context-free, but waiting requires a driver.
    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        {
            let mut state = lock(&self.state);

            if state.fired || Instant::now() >= self.deadline {
                state.fired = true;
                return Poll::Ready(());
            }

            if state
                .waiter
                .as_ref()
                .is_none_or(|old| !old.will_wake(context.waker()))
            {
                state.waiter = Some(context.waker().clone());
            }
        }

        if self.driver.is_none() {
            let handle = Handle::try_current().expect("nosv::time::Sleep polled outside a runtime");
            handle
                .core
                .ensure_running()
                .expect("nosv::time::Sleep polled through a closed or fork-inherited runtime");

            let driver = handle.core.timer.clone();
            driver.register(self.deadline, self.state.clone());
            self.driver = Some(driver);
        }

        Poll::Pending
    }
}

impl Drop for Sleep {
    /// Cancels waiter publication and asks the driver to discard the stale heap entry.
    ///
    /// Cancellation does not remove the record synchronously; its `Arc<TimerState>` keeps all data
    /// alive until the driver observes `cancelled`.
    fn drop(&mut self) {
        {
            let mut state = lock(&self.state);

            state.cancelled = true;
            state.waiter = None;
        }

        if let Some(driver) = &self.driver {
            driver.notify();
        }
    }
}

/// Creates a future that sleeps for at least `duration` on the monotonic clock.
///
/// Excessively large durations that cannot be added to the current [`Instant`] are clamped to an
/// intentionally distant fallback deadline instead of panicking.
pub fn sleep(duration: Duration) -> Sleep {
    sleep_until(
        Instant::now()
            .checked_add(duration)
            .unwrap_or_else(far_future),
    )
}

/// Creates a future that completes at or after a monotonic `deadline`.
///
/// A deadline at or before the first poll completes immediately, even outside a runtime.
pub fn sleep_until(deadline: Instant) -> Sleep {
    Sleep {
        deadline,
        state: Arc::new(Mutex::new(TimerState {
            fired: false,
            cancelled: false,
            waiter: None,
        })),
        driver: None,
    }
}

/// Supplies a representable fallback when adding a requested duration overflows [`Instant`].
///
/// One hundred years is distant enough to preserve the practical meaning of an unrepresentable
/// sleep while still being representable on the supported Linux monotonic clock implementation.
fn far_future() -> Instant {
    Instant::now() + Duration::from_secs(100 * 365 * 24 * 60 * 60)
}

/// Error returned when [`timeout`] reaches its deadline first.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Elapsed;

impl fmt::Display for Elapsed {
    /// Writes a concise description suitable for user-facing error chains.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("deadline elapsed")
    }
}
impl Error for Elapsed {}

/// Future racing an operation against a monotonic timer.
///
/// The operation is polled before the timer on every pass. If both become ready in the same poll,
/// the operation wins. Returning [`Elapsed`] drops the operation future cooperatively; it cannot
/// preempt synchronous work already executing inside that future's `poll`.
pub struct Timeout<F> {
    /// Pinned operation being constrained by the deadline.
    future: Pin<Box<F>>,
    /// Cancellation-safe monotonic timer for the deadline.
    sleep: Sleep,
}
impl<F: Future> Future for Timeout<F> {
    /// Operation output on success, or [`Elapsed`] if the timer wins.
    type Output = Result<F::Output, Elapsed>;

    /// Polls the operation first and then its deadline timer.
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();

        if let Poll::Ready(output) = this.future.as_mut().poll(context) {
            return Poll::Ready(Ok(output));
        }

        if Pin::new(&mut this.sleep).poll(context).is_ready() {
            Poll::Ready(Err(Elapsed))
        } else {
            Poll::Pending
        }
    }
}

/// Races `future` against a monotonic `duration`.
///
/// When the duration elapses first, the future is dropped and [`Elapsed`] is returned. Cancellation
/// safety of the wrapped operation remains the operation's responsibility.
pub fn timeout<F: Future>(duration: Duration, future: F) -> Timeout<F> {
    Timeout {
        future: Box::pin(future),
        sleep: sleep(duration),
    }
}

#[cfg(test)]
#[allow(clippy::missing_docs_in_private_items)]
mod tests {
    use super::*;
    use std::{
        future::Future,
        sync::Arc,
        task::{Context, Wake, Waker},
    };

    struct Noop;
    impl Wake for Noop {
        fn wake(self: Arc<Self>) {}
    }

    #[test]
    fn expired_sleep_is_ready_without_a_runtime() {
        let mut sleep = Box::pin(sleep(Duration::ZERO));
        let waker = Waker::from(Arc::new(Noop));
        let mut context = Context::from_waker(&waker);
        assert_eq!(sleep.as_mut().poll(&mut context), Poll::Ready(()));
    }

    #[test]
    fn ready_operation_wins_a_simultaneous_timeout() {
        let mut timeout = Box::pin(timeout(Duration::ZERO, std::future::ready(42)));
        let waker = Waker::from(Arc::new(Noop));
        let mut context = Context::from_waker(&waker);
        assert_eq!(timeout.as_mut().poll(&mut context), Poll::Ready(Ok(42)));
    }
}
