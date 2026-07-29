//! Spawned tasks, joining, cancellation, current-task access, and yielding.

use crate::{
    affinity::Affinity,
    error::{JoinError, NativeError, SpawnError},
    ffi::{self, RawTask},
    runtime::{Handle, Lifecycle, RuntimeCore, enter_current, weak_handle},
    topology::{CpuId, NumaNodeId},
    util::lock,
};
use std::{
    any::Any,
    future::Future,
    marker::PhantomData,
    panic::{self, AssertUnwindSafe},
    pin::Pin,
    ptr,
    rc::Rc,
    sync::{Arc, Mutex, MutexGuard, Weak},
    task::{Context, Poll, Wake, Waker},
};

/// Stable identifier used for Rust-side diagnostics and registry lookup.
///
/// This identifier is deliberately independent of the native descriptor address. Native
/// descriptor storage can be reused after completion, while a [`TaskId`] remains unambiguous for
/// as long as the corresponding entry is present in the runtime registry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct TaskId(
    /// Monotonically allocated identifier within one runtime generation.
    pub(crate) u64,
);

/// Spawn-time settings collected by [`TaskBuilder`].
///
/// These values are applied before the first native submission. Keeping configuration separate
/// makes post-submission native attribute mutation impossible through the safe API.
#[derive(Default)]
pub(crate) struct TaskConfig {
    /// Optional Rust-only name used in diagnostics.
    pub(crate) name: Option<Box<str>>,
    /// Native scheduler priority to install before submission.
    pub(crate) priority: Option<i32>,
    /// Validated native affinity to install before submission.
    pub(crate) affinity: Option<Affinity>,
    /// Per-task value returned by the shared monitoring-cost callback.
    pub(crate) monitoring_cost: Option<u64>,
}

/// Reason a Rust future stopped being runnable.
///
/// The reason is recorded before user-owned values are dropped. A waker fired by a destructor
/// therefore observes a terminal task and cannot resubmit its descriptor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TerminalKind {
    /// The future returned [`Poll::Ready`].
    Ready,
    /// A cooperative abort request won the race with completion.
    Cancelled,
    /// Polling or destroying the future panicked.
    Panicked,
    /// a nOS-v operation required to continue execution failed.
    RuntimeError,
}

/// Lifecycle of the native descriptor protected by [`NativeGate`].
///
/// Every submit and destroy operation is serialized by the gate mutex. This is the central
/// lifetime guarantee preventing a late waker from submitting a descriptor during destruction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativePhase {
    /// Rust ownership exists, but the native descriptor is not ready for submission.
    Building,
    /// The descriptor may be scheduled or currently executing its callback.
    Live,
    /// Rust polling ended and the completed callback may retire the descriptor.
    Terminal(TerminalKind),
    /// Descriptor destruction is in progress while the gate remains locked.
    Destroying,
    /// The descriptor has been destroyed and can never be submitted again.
    Destroyed,
}

/// State that must be examined atomically when scheduling or retiring a task.
struct NativeGate {
    /// Live native descriptor, removed immediately before destruction.
    task: Option<RawTask>,
    /// Current native ownership and execution phase.
    phase: NativePhase,
    /// Whether this poll epoch already has a native submission in flight.
    wake_submitted: bool,
    /// Whether the run callback is presently polling user code.
    polling: bool,
    /// Whether cooperative cancellation has been requested.
    cancel_requested: bool,
}

/// Type-erased scheduling state shared by callbacks, wakers, and abort handles.
///
/// `TaskCore` intentionally contains no future or output. Stale wakers may retain it after
/// completion, but the terminal gate state leaves them no usable native descriptor.
pub(crate) struct TaskCore {
    /// Authoritative gate for submission, polling transitions, and destruction.
    native: Mutex<NativeGate>,
    /// Runtime that owns the registry and native task type.
    runtime: Weak<RuntimeCore>,
    /// Rust-side registry identifier.
    id: TaskId,
    /// Optional Rust-only diagnostic name.
    name: Option<Box<str>>,
    /// Value exposed through the native monitoring callback.
    monitoring_cost: u64,
}

impl TaskCore {
    /// Locks the native gate, recovering its state after mutex poisoning.
    ///
    /// User panics are caught at poll boundaries, so poisoning is not expected normally. Recovery
    /// still lets cleanup reach a deterministic invariant check rather than unwinding through C.
    fn lock(&self) -> MutexGuard<'_, NativeGate> {
        lock(&self.native)
    }

    /// Coalesces a Rust wake into one ordinary nOS-V submission for this poll epoch.
    ///
    /// The gate remains locked across submission, linearizing native use of the raw descriptor
    /// against destruction. nOS-V's submit/suspend counter closes the wake-before-suspend race.
    fn schedule(&self) {
        let Some(runtime) = self.runtime.upgrade() else {
            return;
        };
        if !runtime.process_matches() {
            return;
        }
        let mut gate = self.lock();
        if gate.phase != NativePhase::Live || gate.wake_submitted {
            return;
        }
        gate.wake_submitted = true;
        let Some(raw) = gate.task else {
            invariant_abort("live task without descriptor");
        };
        if let Err(error) = ffi::submit(raw) {
            eprintln!(
                "nOS-V rejected wake for task {:?} ({:?}): {error}",
                self.id, self.name
            );
            invariant_abort("ordinary submit failed for a live task");
        }
    }

    /// Requests cooperative cancellation and schedules an idle task to observe it.
    ///
    /// Returns `true` only when this call changes a live task from not cancelled to cancelled. A
    /// future already inside `poll` must return before the request can be acted upon.
    pub(crate) fn request_abort(&self) -> bool {
        let Some(runtime) = self.runtime.upgrade() else {
            return false;
        };
        if !runtime.process_matches() {
            return false;
        }
        let mut gate = self.lock();
        if gate.phase != NativePhase::Live || gate.cancel_requested {
            return false;
        }
        gate.cancel_requested = true;
        if !gate.wake_submitted {
            gate.wake_submitted = true;
            let Some(raw) = gate.task else {
                invariant_abort("live task without descriptor");
            };
            if let Err(error) = ffi::submit(raw) {
                eprintln!(
                    "nOS-V rejected cancellation wake for task {:?}: {error}",
                    self.id
                );
                invariant_abort("ordinary cancellation submit failed");
            }
        }
        true
    }

    /// Opens a poll epoch and consumes the submission that entered the callback.
    ///
    /// `false` means cancellation was already pending. This method terminalizes the task in that
    /// case, and the caller must destroy the future without polling it.
    fn start_poll(&self) -> bool {
        let mut gate = self.lock();
        if gate.phase != NativePhase::Live || gate.polling {
            invariant_abort("invalid callback entry state");
        }
        gate.polling = true;
        gate.wake_submitted = false;
        if gate.cancel_requested {
            gate.polling = false;
            gate.phase = NativePhase::Terminal(TerminalKind::Cancelled);
            false
        } else {
            true
        }
    }

    /// Closes a poll epoch that returned [`Poll::Pending`].
    ///
    /// Returns `true` when the callback should suspend. If abort raced the poll, it instead records
    /// terminal cancellation and returns `false`.
    fn finish_pending(&self) -> bool {
        let mut gate = self.lock();
        if gate.phase != NativePhase::Live || !gate.polling {
            invariant_abort("invalid pending state");
        }
        gate.polling = false;
        if gate.cancel_requested {
            gate.phase = NativePhase::Terminal(TerminalKind::Cancelled);
            false
        } else {
            true
        }
    }

    /// Closes a poll epoch with a terminal outcome.
    ///
    /// A cancellation request installed while user code was polling takes precedence over `kind`.
    /// The return value is `true` when the supplied outcome won that race.
    fn finish_terminal(&self, kind: TerminalKind) -> bool {
        let mut gate = self.lock();
        if gate.phase != NativePhase::Live || !gate.polling {
            invariant_abort("invalid terminal state");
        }
        gate.polling = false;
        let cancelled = gate.cancel_requested;
        gate.phase = NativePhase::Terminal(if cancelled {
            TerminalKind::Cancelled
        } else {
            kind
        });
        !cancelled
    }

    /// Replaces a terminal reason without reviving the task.
    ///
    /// Cleanup uses this when a destructor panic is a more accurate outcome than the result first
    /// recorded after polling.
    fn replace_terminal(&self, kind: TerminalKind) {
        let mut gate = self.lock();
        if matches!(gate.phase, NativePhase::Terminal(_)) {
            gate.phase = NativePhase::Terminal(kind);
        }
    }

    /// Converts a still-live pending task into a terminal native-runtime failure.
    ///
    /// An already-terminal phase is retained so cleanup cannot overwrite an earlier decision.
    fn suspend_failed(&self) {
        let mut gate = self.lock();
        if gate.phase == NativePhase::Live {
            gate.phase = NativePhase::Terminal(TerminalKind::RuntimeError);
        }
    }

    /// Destroys a terminal descriptor and permanently closes its scheduling gate.
    ///
    /// Holding the gate through [`ffi::destroy`] means every wake either submits before
    /// destruction starts or observes a non-live phase and becomes a no-op.
    fn retire(&self) -> Result<(), NativeError> {
        let mut gate = self.lock();
        if !matches!(gate.phase, NativePhase::Terminal(_)) {
            return Err(NativeError::InvalidOperation);
        }
        gate.phase = NativePhase::Destroying;
        let raw = gate.task.take().ok_or(NativeError::InvalidOperation)?;
        ffi::destroy(raw)?;
        gate.phase = NativePhase::Destroyed;
        Ok(())
    }
}

impl Wake for TaskCore {
    /// Schedules the task after consuming one strong waker reference.
    fn wake(self: Arc<Self>) {
        self.schedule();
    }
    /// Schedules the task while retaining the caller's strong waker reference.
    fn wake_by_ref(self: &Arc<Self>) {
        self.schedule();
    }
}

/// Type-erased operations needed by C callbacks after the future type is forgotten.
trait ErasedRunnable: Send + Sync {
    /// Polls the future at most once and performs its corresponding state transition.
    fn run_once(&self);
    /// Makes a stored result visible after native descriptor retirement.
    fn publish_native_completion(&self);
    /// Records a panic caught by the callback's outer containment boundary.
    fn force_panic(&self, payload: Box<dyn Any + Send + 'static>);
    /// Records a nOS-v failure caught during callback cleanup.
    fn force_native_error(&self, error: NativeError);
}

/// Allocation whose ownership is transferred to one native descriptor.
///
/// Its pointer is stored in descriptor metadata using unaligned access. The completed callback is
/// the sole successful-path consumer and reconstructs the box after the final run callback ends.
struct NativeOwner {
    /// Type-erased future operations used by native callbacks.
    runnable: Arc<dyn ErasedRunnable>,
    /// Scheduling gate used to retire the descriptor before result publication.
    core: Arc<TaskCore>,
}

/// Shared result cell observed by a single [`JoinHandle`].
struct JoinState<T> {
    /// Result, publication flag, and waiter protected as one synchronization domain.
    inner: Mutex<JoinInner<T>>,
}

/// Mutable contents of [`JoinState`].
struct JoinInner<T> {
    /// Outcome produced by polling but withheld until native completion.
    result: Option<Result<T, JoinError>>,
    /// Whether descriptor retirement is finished and `result` is published.
    native_completed: bool,
    /// Most recent waker supplied by the join task.
    waiter: Option<Waker>,
    /// Whether the sole join result has already been returned.
    consumed: bool,
}

impl<T> JoinState<T> {
    /// Creates an unpublished join cell with no terminal result.
    fn new() -> Self {
        Self {
            inner: Mutex::new(JoinInner {
                result: None,
                native_completed: false,
                waiter: None,
                consumed: false,
            }),
        }
    }
    /// Locks the join cell, recovering its state after mutex poisoning.
    fn lock(&self) -> MutexGuard<'_, JoinInner<T>> {
        lock(&self.inner)
    }
    /// Stores the unique terminal result without publishing it yet.
    ///
    /// Publication is separate so successful joining guarantees that both the future and its C
    /// descriptor have already been retired.
    fn store(&self, result: Result<T, JoinError>) {
        let mut inner = self.lock();
        if inner.result.is_some() {
            invariant_abort("task result stored twice");
        }
        inner.result = Some(result);
    }
    /// Replaces an outcome with a callback-containment error while still unpublished.
    ///
    /// Once native completion is visible, changing the result would race a joiner entitled to
    /// consume it, so late errors are ignored.
    fn replace_error_if_unpublished(&self, error: JoinError) {
        let mut inner = self.lock();
        if !inner.native_completed {
            inner.result = Some(Err(error));
        }
    }
    /// Marks native cleanup complete and wakes the joiner outside the mutex.
    ///
    /// Calling the waker after unlocking avoids re-entrant polling deadlocks. Its invocation is
    /// panic-contained because this method runs beneath an `extern "C"` callback.
    fn publish(&self) {
        let waiter = {
            let mut inner = self.lock();
            if inner.result.is_none() {
                inner.result = Some(Err(JoinError::Runtime(NativeError::InvalidOperation)));
            }
            inner.native_completed = true;
            inner.waiter.take()
        };
        if let Some(waiter) = waiter {
            let _ = panic::catch_unwind(AssertUnwindSafe(|| waiter.wake()));
        }
    }
}

/// Concrete pinned future together with its scheduling and join state.
struct Runnable<F, T> {
    /// Type-independent scheduling state used to construct the poll waker.
    core: Arc<TaskCore>,
    /// Pinned future, removed exactly once before completion publication.
    future: Mutex<Option<Pin<Box<F>>>>,
    /// Destination for the terminal output or error.
    join: Arc<JoinState<T>>,
}

impl<F, T> Runnable<F, T>
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    /// Locks the future slot, recovering it after mutex poisoning.
    ///
    /// Non-parallel native tasks prevent concurrent polling. The mutex additionally makes
    /// ownership explicit to Rust and supports callback cleanup paths.
    fn lock_future(&self) -> MutexGuard<'_, Option<Pin<Box<F>>>> {
        lock(&self.future)
    }

    /// Removes and destroys the future while containing a destructor panic.
    ///
    /// Callers terminalize the gate first, so a future that wakes itself during destruction cannot
    /// resubmit its native descriptor.
    fn take_and_drop_future(&self) -> Option<Box<dyn Any + Send + 'static>> {
        let future = self.lock_future().take();
        panic::catch_unwind(AssertUnwindSafe(|| drop(future))).err()
    }

    /// Destroys an output that lost a completion/cancellation race without unwinding into C.
    fn drop_output(output: T) {
        let _ = panic::catch_unwind(AssertUnwindSafe(|| drop(output)));
    }

    /// Drops the unpolled or pending future and stores cooperative cancellation.
    fn terminal_cancelled(&self) {
        let _ = self.take_and_drop_future();
        self.join.store(Err(JoinError::Cancelled));
    }
}

impl<F, T> ErasedRunnable for Runnable<F, T>
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    /// Performs one future poll from the native run callback.
    ///
    /// Every terminal branch closes the gate before dropping user-owned data. On `Pending`,
    /// `nosv_suspend` is the final meaningful operation so nOS-V's early-wake handshake remains
    /// valid when a wake raced the end of the poll.
    fn run_once(&self) {
        if !self.core.start_poll() {
            self.terminal_cancelled();
            return;
        }
        let Some(handle) = weak_handle(&self.core.runtime) else {
            self.core.finish_terminal(TerminalKind::RuntimeError);
            let _ = self.take_and_drop_future();
            self.join
                .store(Err(JoinError::Runtime(NativeError::InvalidOperation)));
            return;
        };
        let _entered = enter_current(handle);
        let waker = Waker::from(self.core.clone());
        let mut context = Context::from_waker(&waker);
        let polled = {
            let mut future = self.lock_future();
            let Some(future) = future.as_mut() else {
                invariant_abort("future missing during poll");
            };
            panic::catch_unwind(AssertUnwindSafe(|| future.as_mut().poll(&mut context)))
        };

        match polled {
            Ok(Poll::Pending) => {
                if !self.core.finish_pending() {
                    self.terminal_cancelled();
                    return;
                }
                if let Err(error) = ffi::suspend() {
                    self.core.suspend_failed();
                    let _ = self.take_and_drop_future();
                    self.join.store(Err(JoinError::Runtime(error)));
                }
            }
            Ok(Poll::Ready(output)) => {
                if !self.core.finish_terminal(TerminalKind::Ready) {
                    let _ = self.take_and_drop_future();
                    Self::drop_output(output);
                    self.join.store(Err(JoinError::Cancelled));
                    return;
                }
                if let Some(payload) = self.take_and_drop_future() {
                    self.core.replace_terminal(TerminalKind::Panicked);
                    Self::drop_output(output);
                    self.join.store(Err(JoinError::Panic(payload)));
                } else {
                    self.join.store(Ok(output));
                }
            }
            Err(payload) => {
                let panic_wins = self.core.finish_terminal(TerminalKind::Panicked);
                let drop_panic = self.take_and_drop_future();
                if panic_wins {
                    self.join
                        .store(Err(JoinError::Panic(drop_panic.unwrap_or(payload))));
                } else {
                    self.join.store(Err(JoinError::Cancelled));
                }
            }
        }
    }

    /// Publishes the stored outcome after descriptor retirement.
    fn publish_native_completion(&self) {
        self.join.publish();
    }
    /// Converts an outer callback panic into an unpublished panic result.
    fn force_panic(&self, payload: Box<dyn Any + Send + 'static>) {
        self.core.replace_terminal(TerminalKind::Panicked);
        self.join
            .replace_error_if_unpublished(JoinError::Panic(payload));
    }
    /// Converts callback cleanup failure into an unpublished runtime result.
    fn force_native_error(&self, error: NativeError) {
        self.join
            .replace_error_if_unpublished(JoinError::Runtime(error));
    }
}

/// Future resolving only after both Rust state and the native descriptor retire.
///
/// Dropping a join handle detaches the task; it does not cancel it. Use [`JoinHandle::abort`] when
/// cancellation is desired. Awaiting returns an output, cooperative cancellation, a captured panic
/// under unwind-enabled builds, or a native runtime error.
pub struct JoinHandle<T> {
    /// Shared result and waiter state.
    join: Arc<JoinState<T>>,
    /// Scheduling state retained for cancellation.
    core: Arc<TaskCore>,
}

impl<T> JoinHandle<T> {
    /// Requests cooperative, non-preemptive cancellation.
    ///
    /// Returns `true` if this call installed the request. It returns `false` if another abort won,
    /// the task is terminal, the runtime is gone, or this handle was inherited across `fork`. A
    /// future already inside `poll` must return before cancellation takes effect.
    pub fn abort(&self) -> bool {
        self.core.request_abort()
    }
    /// Returns a separately clonable cancellation handle.
    ///
    /// The returned value retains no join output; it only retains the scheduling state required to
    /// request cancellation and observe descriptor retirement.
    pub fn abort_handle(&self) -> AbortHandle {
        AbortHandle {
            core: self.core.clone(),
        }
    }
    /// Reports whether native completion has been published to the join cell.
    ///
    /// A `true` value means polling this handle can complete immediately. This method does not
    /// consume the stored result.
    pub fn is_finished(&self) -> bool {
        self.join.lock().native_completed
    }
}

impl<T> Future for JoinHandle<T> {
    /// The spawned future's output or terminal executor error.
    type Output = Result<T, JoinError>;

    /// Returns a published result or registers the most recent join waker.
    ///
    /// Only one consumer exists because `JoinHandle` is not clonable. Equivalent wakers are not
    /// replaced, avoiding unnecessary reference-count traffic.
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let mut inner = self.join.lock();
        assert!(!inner.consumed, "JoinHandle polled after completion");
        if inner.native_completed {
            inner.consumed = true;
            Poll::Ready(
                inner
                    .result
                    .take()
                    .unwrap_or(Err(JoinError::Runtime(NativeError::InvalidOperation))),
            )
        } else {
            if inner
                .waiter
                .as_ref()
                .is_none_or(|old| !old.will_wake(context.waker()))
            {
                inner.waiter = Some(context.waker().clone());
            }
            Poll::Pending
        }
    }
}

/// A clonable cooperative-cancellation capability.
///
/// Unlike [`JoinHandle`], this carries no output and may be shared freely. It lets code distribute
/// cancellation authority without also granting permission to consume the task result.
#[derive(Clone)]
pub struct AbortHandle {
    /// Scheduling state used to linearize cancellation with completion.
    core: Arc<TaskCore>,
}
impl AbortHandle {
    /// Requests cancellation, returning whether this call won the request race.
    ///
    /// Cancellation is cooperative: an actively polling future is not interrupted, and shutdown
    /// can wait indefinitely for a future that never returns from `poll`.
    pub fn abort(&self) -> bool {
        self.core.request_abort()
    }
    /// Reports whether the native descriptor has been destroyed.
    ///
    /// This is an advisory snapshot; another thread may complete the task immediately afterward.
    pub fn is_finished(&self) -> bool {
        self.core.lock().phase == NativePhase::Destroyed
    }
}

/// Configuration fixed before a spawned task's first submission.
///
/// The builder is consumed by [`TaskBuilder::spawn`], making post-submission mutation impossible
/// through the safe API. It borrows a [`Handle`] only while settings are assembled.
pub struct TaskBuilder<'a> {
    /// Runtime on which the configured task will be created.
    handle: &'a Handle,
    /// Settings accumulated before descriptor creation.
    config: TaskConfig,
}
impl<'a> TaskBuilder<'a> {
    /// Starts a builder with native defaults and a monitoring cost of one.
    pub(crate) fn new(handle: &'a Handle) -> Self {
        Self {
            handle,
            config: TaskConfig::default(),
        }
    }
    /// Adds a Rust-only diagnostic name.
    ///
    /// The name does not become a distinct native task type. Current nOS-V task-type destruction
    /// is a no-op, so per-task native labels would accumulate for the process lifetime.
    pub fn rust_name(mut self, name: impl Into<Box<str>>) -> Self {
        self.config.name = Some(name.into());
        self
    }
    /// Sets native scheduler priority before first submission.
    ///
    /// The integer is forwarded unchanged; its interpretation follows the active nOS-V scheduling
    /// policy and configuration.
    pub fn priority(mut self, priority: i32) -> Self {
        self.config.priority = Some(priority);
        self
    }
    /// Sets validated native affinity before first submission.
    ///
    /// Conversion to the packed C bitfield happens before descriptor creation, so validation
    /// failure cannot strand partially initialized native ownership.
    pub fn affinity(mut self, affinity: Affinity) -> Self {
        self.config.affinity = Some(affinity);
        self
    }
    /// Sets the value returned by the shared monitoring-cost callback.
    ///
    /// This is monitoring metadata, not scheduler weight. A shared native task type can expose a
    /// per-task value by reading the Rust owner stored in descriptor metadata.
    pub fn monitoring_cost(mut self, cost: u64) -> Self {
        self.config.monitoring_cost = Some(cost);
        self
    }
    /// Creates and submits the configured future.
    ///
    /// Spawn is linearized with shutdown: it either registers a submitted descriptor that shutdown
    /// will drain, or fails without publishing a live native task.
    ///
    /// # Errors
    ///
    /// Returns an error after shutdown, in a forked child, for invalid affinity, or when native
    /// descriptor creation, metadata access, or initial submission fails.
    pub fn spawn<F, T>(self, future: F) -> Result<JoinHandle<T>, SpawnError>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        spawn_on(self.handle, self.config, future)
    }
}

/// Spawns a future on the runtime currently polling this task.
///
/// A current handle exists only while a root or spawned future is being polled. Code with an
/// explicit [`Handle`] should prefer [`Handle::spawn`].
///
/// # Errors
///
/// Returns [`SpawnError::RuntimeClosed`] outside a runtime poll context, or forwards the normal
/// validation and native errors from [`Handle::spawn`].
pub fn spawn<F, T>(future: F) -> Result<JoinHandle<T>, SpawnError>
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    Handle::try_current()
        .map_err(SpawnError::from)?
        .spawn(future)
}

/// Constructs, registers, and initially submits a native task for one future.
///
/// The runtime-state lock remains held from the lifecycle check through submission. It acts as a
/// spawn permit: shutdown cannot drain until the task is either registered or fully reclaimed.
pub(crate) fn spawn_on<F, T>(
    handle: &Handle,
    config: TaskConfig,
    future: F,
) -> Result<JoinHandle<T>, SpawnError>
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    if !handle.core.process_matches() {
        return Err(SpawnError::ForkedProcess);
    }
    let raw_affinity = config.affinity.map(Affinity::to_raw).transpose()?;
    let mut state = handle.core.lock_state();
    if state.lifecycle != Lifecycle::Running {
        return Err(SpawnError::RuntimeClosed);
    }

    let id = handle.core.next_task_id();
    let join = Arc::new(JoinState::new());
    let core = Arc::new(TaskCore {
        native: Mutex::new(NativeGate {
            task: None,
            phase: NativePhase::Building,
            wake_submitted: false,
            polling: false,
            cancel_requested: false,
        }),
        runtime: Arc::downgrade(&handle.core),
        id,
        name: config.name,
        monitoring_cost: config.monitoring_cost.unwrap_or(1),
    });
    let runnable: Arc<dyn ErasedRunnable> = Arc::new(Runnable {
        core: core.clone(),
        future: Mutex::new(Some(Box::pin(future))),
        join: join.clone(),
    });
    let owner = Box::new(NativeOwner {
        runnable,
        core: core.clone(),
    });
    let raw = ffi::create(
        handle.core.task_type,
        std::mem::size_of::<*mut NativeOwner>(),
    )?;
    let metadata = match ffi::metadata(raw) {
        Ok(metadata) => metadata,
        Err(error) => {
            let _ = ffi::destroy(raw);
            return Err(SpawnError::Native(error));
        }
    };
    let owner_pointer = Box::into_raw(owner);
    // SAFETY: metadata has pointer size, may be unaligned, and uniquely belongs
    // to this unsubmitted descriptor. Completed callback is its sole consumer.
    unsafe { ptr::write_unaligned(metadata.as_ptr().cast::<*mut NativeOwner>(), owner_pointer) };
    if let Some(priority) = config.priority {
        ffi::set_priority(raw, priority);
    }
    if let Some(affinity) = raw_affinity {
        ffi::set_affinity(raw, affinity);
    }
    {
        let mut gate = core.lock();
        gate.task = Some(raw);
        gate.phase = NativePhase::Live;
        gate.wake_submitted = true;
    }
    state.tasks.insert(id, core.clone());
    let submitted = {
        let gate = core.lock();
        ffi::submit(gate.task.expect("live descriptor"))
    };
    if let Err(error) = submitted {
        state.tasks.remove(&id);
        let mut gate = core.lock();
        gate.phase = NativePhase::Destroying;
        let raw = gate.task.take().expect("created descriptor");
        let _ = ffi::destroy(raw);
        gate.phase = NativePhase::Destroyed;
        // SAFETY: submission failed, so the completed callback cannot consume it.
        unsafe { drop(Box::from_raw(owner_pointer)) };
        return Err(SpawnError::Native(error));
    }
    Ok(JoinHandle { join, core })
}

/// Reads the unaligned owner pointer stored in native task metadata.
///
/// # Safety
///
/// `raw` must have been created by [`spawn_on`], its metadata must be initialized, and its completed
/// callback must not have consumed the [`NativeOwner`]. Only that callback—or the proven
/// unsubmitted failure path—may reconstruct the returned pointer as a `Box`.
unsafe fn owner_pointer(raw: RawTask) -> *mut NativeOwner {
    let metadata = ffi::metadata(raw).unwrap_or_else(|_| invariant_abort("task metadata missing"));
    // SAFETY: spawn_on wrote this pointer with write_unaligned and ownership has
    // not yet been consumed by the completed callback.
    unsafe { ptr::read_unaligned(metadata.as_ptr().cast::<*mut NativeOwner>()) }
}

/// nOS-V run callback that polls the Rust future exactly once.
///
/// The whole callback and the erased poll are panic-contained. An unexpected panic that cannot be
/// represented as a join error aborts instead of unwinding into C.
///
/// # Safety
///
/// nOS-V must supply the live, non-parallel descriptor whose metadata contains an initialized
/// [`NativeOwner`] created by [`spawn_on`].
pub(crate) unsafe extern "C" fn run_callback(pointer: nosv_sys::nosv_task_t) {
    let callback = panic::catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: nOS-V invokes callbacks with the live task descriptor.
        let raw = unsafe { RawTask::from_ptr(pointer) }
            .unwrap_or_else(|| invariant_abort("null callback task"));
        // SAFETY: descriptor owns metadata until completion.
        let owner = unsafe { owner_pointer(raw).as_ref() }
            .unwrap_or_else(|| invariant_abort("null task owner"));
        let runnable = owner.runnable.clone();
        if let Err(payload) = panic::catch_unwind(AssertUnwindSafe(|| runnable.run_once())) {
            runnable.force_panic(payload);
        }
    }));
    if callback.is_err() {
        invariant_abort("panic in task run callback");
    }
}

/// Retires a completed native descriptor and publishes its Rust join result.
///
/// This is the unique successful-path consumer of [`NativeOwner`]. It destroys the C descriptor
/// under the scheduling gate before waking the joiner, then removes the registry entry. Uncertain
/// native ownership is leaked before aborting rather than risking use-after-free.
///
/// # Safety
///
/// nOS-V must invoke this exactly once for a terminal descriptor created by [`spawn_on`], after its
/// final run callback returns and while descriptor metadata remains accessible.
pub(crate) unsafe extern "C" fn completed_callback(pointer: nosv_sys::nosv_task_t) {
    let completed = panic::catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: nOS-V invokes this exactly once with a live terminal descriptor.
        let raw = unsafe { RawTask::from_ptr(pointer) }
            .unwrap_or_else(|| invariant_abort("null completed task"));
        // SAFETY: this callback is the sole consumer of NativeOwner.
        let pointer = unsafe { owner_pointer(raw) };
        // SAFETY: spawn_on allocated this owner with Box::into_raw.
        let owner = unsafe { Box::from_raw(pointer) };
        if let Err(error) = owner.core.retire() {
            owner.runnable.force_native_error(error);
            std::mem::forget(owner);
            invariant_abort("could not retire completed native task");
        }
        let runtime = owner.core.runtime.upgrade();
        let id = owner.core.id;
        owner.runnable.publish_native_completion();
        let _ = panic::catch_unwind(AssertUnwindSafe(|| drop(owner)));
        if let Some(runtime) = runtime {
            runtime.task_completed(id);
        }
    }));
    if completed.is_err() {
        invariant_abort("panic in completed callback");
    }
}

/// Returns a task's Rust-side monitoring cost to nOS-V.
///
/// Panic containment falls back to one. The value is observational metadata and does not
/// participate in ownership or scheduling state transitions.
///
/// # Safety
///
/// nOS-V must supply a live descriptor initialized by [`spawn_on`], and its [`NativeOwner`] must
/// remain allocated for this call.
pub(crate) unsafe extern "C" fn cost_callback(pointer: nosv_sys::nosv_task_t) -> u64 {
    panic::catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: cost callback receives a live descriptor with initialized metadata.
        let raw = unsafe { RawTask::from_ptr(pointer) }
            .unwrap_or_else(|| invariant_abort("null cost task"));
        // SAFETY: NativeOwner remains live until the completed callback.
        unsafe { owner_pointer(raw).as_ref() }.map_or(1, |owner| owner.core.monitoring_cost)
    }))
    .unwrap_or(1)
}

/// Reports an executor invariant violation and terminates the process.
///
/// Continuing after an impossible ownership transition could submit or free a stale raw pointer.
/// Aborting preserves memory safety when native descriptor ownership can no longer be proven.
fn invariant_abort(message: &str) -> ! {
    eprintln!("nOS-V Rust runtime invariant failed: {message}");
    std::process::abort()
}

/// Scoped access to queries valid only in a current nOS-V task.
///
/// The higher-ranked callback in [`with_current`] prevents this value from escaping. Its `Rc`
/// marker also makes the capability neither `Send` nor `Sync`, so it cannot move to another worker.
pub struct CurrentTask<'a> {
    /// Invariant lifetime tying the capability to one closure invocation.
    _scope: PhantomData<&'a mut ()>,
    /// Marker preventing transfer or sharing across threads.
    _not_send: PhantomData<Rc<()>>,
}
impl CurrentTask<'_> {
    /// Returns the system CPU currently executing this task.
    ///
    /// This is a snapshot. A suspended task can resume on a different nOS-V worker pthread.
    pub fn current_cpu(&self) -> Result<CpuId, NativeError> {
        CpuId::from_native(ffi::current_cpu()?)
    }
    /// Returns the system NUMA node currently executing this task.
    ///
    /// The result describes the present worker and is not a lasting affinity guarantee.
    pub fn current_numa_node(&self) -> Result<NumaNodeId, NativeError> {
        NumaNodeId::from_native(ffi::current_numa_node()?)
    }
}

/// Runs a query closure only while a nOS-v task context exists.
///
/// The limited capability exposes transient topology queries without exposing raw task handles or
/// stackful native blocking operations to ordinary async code.
///
/// # Errors
///
/// Returns [`NativeError::OutsideTask`] when no current native task exists.
pub fn with_current<R>(
    query: impl for<'a> FnOnce(&CurrentTask<'a>) -> Result<R, NativeError>,
) -> Result<R, NativeError> {
    if ffi::current().is_none() {
        return Err(NativeError::OutsideTask);
    }
    query(&CurrentTask {
        _scope: PhantomData,
        _not_send: PhantomData,
    })
}

/// A future that cooperatively yields one scheduling turn.
///
/// Its first poll self-wakes and returns [`Poll::Pending`], causing native suspension and
/// resubmission. The second poll completes. This is a scheduling hint, not a fairness guarantee.
#[derive(Debug, Default)]
pub struct YieldNow {
    /// Whether the required pending result has already been produced.
    yielded: bool,
}
impl Future for YieldNow {
    /// Yielding completes without producing a value.
    type Output = ();

    /// Self-wakes and yields on the first poll, then completes on the second.
    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
        if self.yielded {
            Poll::Ready(())
        } else {
            self.yielded = true;
            context.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

/// Creates a future that cooperatively yields the current async task once.
///
/// The future must be awaited to have any effect. It can add pending points to CPU-heavy async
/// loops, but it does not make blocking or unbounded work preemptible.
pub fn yield_now() -> YieldNow {
    YieldNow { yielded: false }
}
