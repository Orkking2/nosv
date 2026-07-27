//! Cancellation-safe monotonic timers driven by one nOS-V task per runtime.

use crate::{ffi, runtime::Handle};
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
        Arc, Condvar, Mutex, MutexGuard,
        atomic::{AtomicU64, Ordering as AtomicOrdering},
    },
    task::{Context, Poll, Waker},
    time::{Duration, Instant},
};

struct TimerState {
    fired: bool,
    cancelled: bool,
    waiter: Option<Waker>,
}

struct TimerEntry {
    deadline: Instant,
    id: u64,
    state: Arc<Mutex<TimerState>>,
}

impl PartialEq for TimerEntry {
    fn eq(&self, other: &Self) -> bool {
        self.deadline == other.deadline && self.id == other.id
    }
}
impl Eq for TimerEntry {}
impl PartialOrd for TimerEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TimerEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .deadline
            .cmp(&self.deadline)
            .then_with(|| other.id.cmp(&self.id))
    }
}

struct DriverData {
    timers: BinaryHeap<TimerEntry>,
    shutdown: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DriverPark {
    Building,
    Running,
    Paused,
    DeadlineWait,
    WakePending,
    Terminal,
    Destroyed,
}

struct DriverGate {
    raw: Option<ffi::RawTask>,
    phase: DriverPark,
    notified: bool,
}

pub(crate) struct TimerDriver {
    data: Mutex<DriverData>,
    gate: Mutex<DriverGate>,
    next_id: AtomicU64,
    completed: Mutex<bool>,
    completed_cv: Condvar,
}

impl TimerDriver {
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
        let raw = ffi::create(task_type, std::mem::size_of::<*mut Arc<Self>>())?;
        let metadata = match ffi::metadata(raw) {
            Ok(metadata) => metadata,
            Err(error) => {
                let _ = ffi::destroy(raw);
                return Err(error);
            }
        };
        let owner = Box::into_raw(Box::new(driver.clone()));
        // SAFETY: the metadata is pointer-sized but may be unaligned. The
        // completed callback is the sole consumer of this Box pointer.
        unsafe { ptr::write_unaligned(metadata.as_ptr().cast::<*mut Arc<Self>>(), owner) };
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
            unsafe { drop(Box::from_raw(owner)) };
            return Err(error);
        }
        Ok(driver)
    }

    fn lock_data(&self) -> MutexGuard<'_, DriverData> {
        self.data.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn lock_gate(&self) -> MutexGuard<'_, DriverGate> {
        self.gate.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn register(&self, deadline: Instant, state: Arc<Mutex<TimerState>>) {
        let id = self.next_id.fetch_add(1, AtomicOrdering::Relaxed);
        if id == u64::MAX {
            invariant_abort("timer identifier exhausted");
        }
        let mut data = self.lock_data();
        if data.shutdown {
            let mut state = state.lock().unwrap_or_else(|p| p.into_inner());
            state.cancelled = true;
            state.waiter = None;
            return;
        }
        data.timers.push(TimerEntry {
            deadline,
            id,
            state,
        });
        drop(data);
        self.notify();
    }

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

    fn run(&self) {
        loop {
            let (wait, wake) = {
                let mut data = self.lock_data();
                if data.shutdown {
                    for entry in data.timers.drain() {
                        let mut state = entry.state.lock().unwrap_or_else(|p| p.into_inner());
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
                        let mut state = entry.state.lock().unwrap_or_else(|p| p.into_inner());
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
                            .unwrap_or_else(|p| p.into_inner())
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

    pub(crate) fn shutdown_and_wait(&self) {
        self.lock_data().shutdown = true;
        self.notify();
        let mut completed = self.completed.lock().unwrap_or_else(|p| p.into_inner());
        while !*completed {
            completed = self
                .completed_cv
                .wait(completed)
                .unwrap_or_else(|p| p.into_inner());
        }
    }

    fn retire(&self) {
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
        drop(gate);
        *self.completed.lock().unwrap_or_else(|p| p.into_inner()) = true;
        self.completed_cv.notify_all();
    }
}

unsafe fn timer_owner(raw: ffi::RawTask) -> *mut Arc<TimerDriver> {
    let metadata = ffi::metadata(raw).unwrap_or_else(|_| invariant_abort("timer metadata missing"));
    // SAFETY: start wrote this Box pointer with write_unaligned.
    unsafe { ptr::read_unaligned(metadata.as_ptr().cast::<*mut Arc<TimerDriver>>()) }
}

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

pub(crate) unsafe extern "C" fn completed_callback(pointer: nosv_sys::nosv_task_t) {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: nOS-V supplies the terminal timer descriptor.
        let raw = unsafe { ffi::RawTask::from_ptr(pointer) }
            .unwrap_or_else(|| invariant_abort("null completed timer"));
        // SAFETY: this callback is the sole consumer of the Box pointer.
        let owner = unsafe { Box::from_raw(timer_owner(raw)) };
        owner.retire();
        drop(owner);
    }));
    if result.is_err() {
        invariant_abort("panic in timer completed callback");
    }
}

fn invariant_abort(message: &str) -> ! {
    eprintln!("nOS-V timer invariant failed: {message}");
    std::process::abort()
}

/// A monotonic deadline future.
pub struct Sleep {
    deadline: Instant,
    state: Arc<Mutex<TimerState>>,
    driver: Option<Arc<TimerDriver>>,
}

impl Sleep {
    /// Returns this timer's monotonic deadline.
    pub const fn deadline(&self) -> Instant {
        self.deadline
    }
}

impl Future for Sleep {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
        {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
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
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.cancelled = true;
        state.waiter = None;
        drop(state);
        if let Some(driver) = &self.driver {
            driver.notify();
        }
    }
}

/// Sleeps for at least `duration` according to the monotonic clock.
pub fn sleep(duration: Duration) -> Sleep {
    sleep_until(
        Instant::now()
            .checked_add(duration)
            .unwrap_or_else(far_future),
    )
}

/// Sleeps until a monotonic deadline.
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

fn far_future() -> Instant {
    Instant::now() + Duration::from_secs(100 * 365 * 24 * 60 * 60)
}

/// Error returned when [`timeout`] reaches its deadline first.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Elapsed;
impl fmt::Display for Elapsed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("deadline elapsed")
    }
}
impl Error for Elapsed {}

/// Future racing an operation against a monotonic timer.
pub struct Timeout<F> {
    future: Pin<Box<F>>,
    sleep: Sleep,
}
impl<F: Future> Future for Timeout<F> {
    type Output = Result<F::Output, Elapsed>;
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

/// Races `future` against `duration`.
pub fn timeout<F: Future>(duration: Duration, future: F) -> Timeout<F> {
    Timeout {
        future: Box::pin(future),
        sleep: sleep(duration),
    }
}
