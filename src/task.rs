//! Spawned tasks, joining, cancellation, current-task access, and yielding.

use crate::{
    affinity::Affinity,
    error::{JoinError, NativeError, SpawnError},
    ffi::{self, RawTask},
    runtime::{Handle, Lifecycle, RuntimeCore, enter_current, weak_handle},
    topology::{CpuId, NumaNodeId},
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

/// Stable identifier used only for Rust-side diagnostics and registry lookup.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct TaskId(pub(crate) u64);

#[derive(Default)]
pub(crate) struct TaskConfig {
    pub(crate) name: Option<Box<str>>,
    pub(crate) priority: Option<i32>,
    pub(crate) affinity: Option<Affinity>,
    pub(crate) monitoring_cost: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TerminalKind {
    Ready,
    Cancelled,
    Panicked,
    RuntimeError,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativePhase {
    Building,
    Live,
    Terminal(TerminalKind),
    Destroying,
    Destroyed,
}

struct NativeGate {
    task: Option<RawTask>,
    phase: NativePhase,
    wake_submitted: bool,
    polling: bool,
    cancel_requested: bool,
}

pub(crate) struct TaskCore {
    native: Mutex<NativeGate>,
    runtime: Weak<RuntimeCore>,
    id: TaskId,
    name: Option<Box<str>>,
    monitoring_cost: u64,
}

impl TaskCore {
    fn lock(&self) -> MutexGuard<'_, NativeGate> {
        self.native
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

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

    fn replace_terminal(&self, kind: TerminalKind) {
        let mut gate = self.lock();
        if matches!(gate.phase, NativePhase::Terminal(_)) {
            gate.phase = NativePhase::Terminal(kind);
        }
    }

    fn suspend_failed(&self) {
        let mut gate = self.lock();
        if gate.phase == NativePhase::Live {
            gate.phase = NativePhase::Terminal(TerminalKind::RuntimeError);
        }
    }

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
    fn wake(self: Arc<Self>) {
        self.schedule();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.schedule();
    }
}

trait ErasedRunnable: Send + Sync {
    fn run_once(&self);
    fn publish_native_completion(&self);
    fn force_panic(&self, payload: Box<dyn Any + Send + 'static>);
    fn force_native_error(&self, error: NativeError);
}

struct NativeOwner {
    runnable: Arc<dyn ErasedRunnable>,
    core: Arc<TaskCore>,
}

struct JoinState<T> {
    inner: Mutex<JoinInner<T>>,
}
struct JoinInner<T> {
    result: Option<Result<T, JoinError>>,
    native_completed: bool,
    waiter: Option<Waker>,
    consumed: bool,
}

impl<T> JoinState<T> {
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
    fn lock(&self) -> MutexGuard<'_, JoinInner<T>> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }
    fn store(&self, result: Result<T, JoinError>) {
        let mut inner = self.lock();
        if inner.result.is_some() {
            invariant_abort("task result stored twice");
        }
        inner.result = Some(result);
    }
    fn replace_error_if_unpublished(&self, error: JoinError) {
        let mut inner = self.lock();
        if !inner.native_completed {
            inner.result = Some(Err(error));
        }
    }
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

struct Runnable<F, T> {
    core: Arc<TaskCore>,
    future: Mutex<Option<Pin<Box<F>>>>,
    join: Arc<JoinState<T>>,
}

impl<F, T> Runnable<F, T>
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    fn lock_future(&self) -> MutexGuard<'_, Option<Pin<Box<F>>>> {
        self.future.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn take_and_drop_future(&self) -> Option<Box<dyn Any + Send + 'static>> {
        let future = self.lock_future().take();
        panic::catch_unwind(AssertUnwindSafe(|| drop(future))).err()
    }

    fn drop_output(output: T) {
        let _ = panic::catch_unwind(AssertUnwindSafe(|| drop(output)));
    }

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

    fn publish_native_completion(&self) {
        self.join.publish();
    }
    fn force_panic(&self, payload: Box<dyn Any + Send + 'static>) {
        self.core.replace_terminal(TerminalKind::Panicked);
        self.join
            .replace_error_if_unpublished(JoinError::Panic(payload));
    }
    fn force_native_error(&self, error: NativeError) {
        self.join
            .replace_error_if_unpublished(JoinError::Runtime(error));
    }
}

/// Future resolving only after both Rust state and the native descriptor retire.
pub struct JoinHandle<T> {
    join: Arc<JoinState<T>>,
    core: Arc<TaskCore>,
}

impl<T> JoinHandle<T> {
    /// Requests cooperative cancellation.
    pub fn abort(&self) -> bool {
        self.core.request_abort()
    }
    /// Returns a separately clonable cancellation handle.
    pub fn abort_handle(&self) -> AbortHandle {
        AbortHandle {
            core: self.core.clone(),
        }
    }
    /// Returns whether native completion has been published.
    pub fn is_finished(&self) -> bool {
        self.join.lock().native_completed
    }
}

impl<T> Future for JoinHandle<T> {
    type Output = Result<T, JoinError>;
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
#[derive(Clone)]
pub struct AbortHandle {
    core: Arc<TaskCore>,
}
impl AbortHandle {
    /// Requests cancellation, returning whether this call won the request race.
    pub fn abort(&self) -> bool {
        self.core.request_abort()
    }
    /// Returns whether native completion has been published.
    pub fn is_finished(&self) -> bool {
        self.core.lock().phase == NativePhase::Destroyed
    }
}

/// Configuration fixed before a spawned task's first submission.
pub struct TaskBuilder<'a> {
    handle: &'a Handle,
    config: TaskConfig,
}
impl<'a> TaskBuilder<'a> {
    pub(crate) fn new(handle: &'a Handle) -> Self {
        Self {
            handle,
            config: TaskConfig::default(),
        }
    }
    /// Adds a Rust-only diagnostic name.
    pub fn rust_name(mut self, name: impl Into<Box<str>>) -> Self {
        self.config.name = Some(name.into());
        self
    }
    /// Sets native priority before submission.
    pub fn priority(mut self, priority: i32) -> Self {
        self.config.priority = Some(priority);
        self
    }
    /// Sets native affinity before submission.
    pub fn affinity(mut self, affinity: Affinity) -> Self {
        self.config.affinity = Some(affinity);
        self
    }
    /// Sets the value returned by the type's monitoring cost callback.
    pub fn monitoring_cost(mut self, cost: u64) -> Self {
        self.config.monitoring_cost = Some(cost);
        self
    }
    /// Creates and submits the configured future.
    pub fn spawn<F, T>(self, future: F) -> Result<JoinHandle<T>, SpawnError>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        spawn_on(self.handle, self.config, future)
    }
}

/// Spawns on the runtime currently polling this task.
pub fn spawn<F, T>(future: F) -> Result<JoinHandle<T>, SpawnError>
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    Handle::try_current()
        .map_err(|_| SpawnError::RuntimeClosed)?
        .spawn(future)
}

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

unsafe fn owner_pointer(raw: RawTask) -> *mut NativeOwner {
    let metadata = ffi::metadata(raw).unwrap_or_else(|_| invariant_abort("task metadata missing"));
    // SAFETY: spawn_on wrote this pointer with write_unaligned and ownership has
    // not yet been consumed by the completed callback.
    unsafe { ptr::read_unaligned(metadata.as_ptr().cast::<*mut NativeOwner>()) }
}

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

fn invariant_abort(message: &str) -> ! {
    eprintln!("nOS-V Rust runtime invariant failed: {message}");
    std::process::abort()
}

/// Scoped access to queries valid only in a current nOS-V task.
pub struct CurrentTask<'a> {
    _scope: PhantomData<&'a mut ()>,
    _not_send: PhantomData<Rc<()>>,
}
impl CurrentTask<'_> {
    /// Returns the system CPU currently executing this task.
    pub fn current_cpu(&self) -> Result<CpuId, NativeError> {
        CpuId::from_native(ffi::current_cpu()?)
    }
    /// Returns the system NUMA node currently executing this task.
    pub fn current_numa_node(&self) -> Result<NumaNodeId, NativeError> {
        NumaNodeId::from_native(ffi::current_numa_node()?)
    }
}

/// Runs a query closure only when an nOS-V task context exists.
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

/// A future that yields one scheduling turn.
#[derive(Debug, Default)]
pub struct YieldNow {
    yielded: bool,
}
impl Future for YieldNow {
    type Output = ();
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

/// Cooperatively yields this future once.
pub fn yield_now() -> YieldNow {
    YieldNow { yielded: false }
}
