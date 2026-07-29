//! Runtime construction, lifecycle, handles, and attached-thread `block_on`.
//!
//! The runtime owns native task types and drivers, while cloneable handles own
//! only an `Arc` to Rust state. Initialization and shutdown remain on one owner
//! thread because nOS-V tracks per-thread reference counts. Spawn and query
//! operations use the shared lifecycle mutex to linearize with shutdown.

use crate::{
    error::{BlockOnError, InitError, NativeError, RuntimeClosed, ShutdownError, SpawnError},
    ffi::{self, RawTaskType},
    memory::MemoryStats,
    task::{self, JoinHandle, TaskBuilder, TaskCore, TaskId},
    topology::Topology,
    util::lock,
};
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    future::Future,
    marker::PhantomData,
    panic::{self, AssertUnwindSafe},
    pin::pin,
    rc::Rc,
    sync::{
        Arc, Condvar, Mutex, MutexGuard, Weak,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
    task::{Context, Poll, Wake, Waker},
    thread::{self, ThreadId},
};

thread_local! {
    /// Handle visible only while this pthread is polling a runtime future.
    ///
    /// A scoped guard restores the prior value, including during unwinding.
    static CURRENT: RefCell<Option<Handle>> = const { RefCell::new(None) };
    /// Prevents recursive attachment by nested `block_on` calls on one thread.
    static IN_BLOCK_ON: Cell<bool> = const { Cell::new(false) };
}

/// Runtime lifecycle protected by `RuntimeCore::state`.
///
/// The state lock is the linearization point for spawn, query capabilities, and
/// the transition that closes the runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Lifecycle {
    /// New tasks and native queries are accepted.
    Running,
    /// New work is rejected while registered tasks are cooperatively drained.
    Closing,
    /// Drivers and task types have retired and native shutdown has completed.
    Closed,
}

/// Mutable registry state shared by handles and native completion callbacks.
///
/// Keeping lifecycle and task membership under one mutex ensures a spawn either
/// becomes registered before shutdown or observes the closing state.
pub(crate) struct RuntimeState {
    /// Current acceptance and shutdown phase.
    pub(crate) lifecycle: Lifecycle,
    /// Strong references keeping every submitted task registered until its
    /// completed callback has destroyed native ownership.
    pub(crate) tasks: HashMap<TaskId, Arc<TaskCore>>,
}

/// Thread-safe state shared by `Runtime`, `Handle`, tasks, and drivers.
///
/// This object contains no public raw handles. Its recorded PID makes all
/// capability paths fail closed after `fork`, and its owner thread is used only
/// by the non-`Send` `Runtime` during attach and shutdown.
pub(crate) struct RuntimeCore {
    /// Thread that balanced native initialization and must perform shutdown.
    pub(crate) owner_thread: ThreadId,
    /// Process generation in which native descriptors were created.
    pub(crate) pid: libc::pid_t,
    /// Native type shared by all Rust async tasks in this runtime.
    pub(crate) task_type: RawTaskType,
    #[cfg(feature = "time")]
    /// Native type dedicated to the stackful timer driver.
    pub(crate) timer_type: RawTaskType,
    #[cfg(feature = "time")]
    /// Single timer driver and command heap for this runtime.
    pub(crate) timer: Arc<crate::time::TimerDriver>,
    /// Authoritative lifecycle and task-registry mutex.
    pub(crate) state: Mutex<RuntimeState>,
    /// Wakes shutdown after the final registered task callback has finished.
    pub(crate) drained: Condvar,
    /// Monotonic source of collision-free Rust-side task identifiers.
    next_task_id: AtomicU64,
}

impl RuntimeCore {
    /// Locks lifecycle and registry state, recovering data from poison.
    ///
    /// Panics are already contained at C callback boundaries, so poisoning is a
    /// diagnostic artifact rather than evidence that the native pointer is safe
    /// to abandon. Recovering preserves the ability to cancel and drain tasks.
    pub(crate) fn lock_state(&self) -> MutexGuard<'_, RuntimeState> {
        lock(&self.state)
    }

    /// Reports whether the caller is in the process that created this core.
    ///
    /// This read-only PID comparison is performed before any fork-sensitive FFI
    /// operation. A false result must cause the calling path to make no nOS-V
    /// calls through inherited descriptors.
    pub(crate) fn process_matches(&self) -> bool {
        // SAFETY: getpid has no preconditions.
        self.pid == unsafe { libc::getpid() }
    }

    /// Allocates the next Rust registry identifier.
    ///
    /// Exhaustion aborts rather than wrapping: a collision could replace a live
    /// registry entry and let shutdown proceed while its native descriptor still
    /// exists, which is a safety failure rather than a recoverable resource error.
    pub(crate) fn next_task_id(&self) -> TaskId {
        let id = self.next_task_id.fetch_add(1, Relaxed);
        if id == u64::MAX {
            eprintln!("nOS-V Rust runtime exhausted its task identifier space");
            std::process::abort();
        }
        TaskId(id)
    }

    /// Removes a fully retired task and notifies shutdown when the registry empties.
    ///
    /// Native completed callbacks invoke this only after result publication and
    /// owner destruction, so an empty registry means no task-owned Rust cleanup
    /// remains before driver and type teardown.
    pub(crate) fn task_completed(&self, id: TaskId) {
        let mut state = self.lock_state();
        state.tasks.remove(&id);
        if state.tasks.is_empty() {
            self.drained.notify_all();
        }
    }

    /// Validates the process generation and open lifecycle for capability access.
    ///
    /// This compact check is suitable where another invariant already prevents
    /// shutdown from invalidating native state during the subsequent operation.
    /// Queries without such an invariant hold the state lock through their FFI.
    pub(crate) fn ensure_running(&self) -> Result<(), RuntimeClosed> {
        if !self.process_matches() || self.lock_state().lifecycle != Lifecycle::Running {
            Err(RuntimeClosed)
        } else {
            Ok(())
        }
    }
}

/// Builder for Rust-layer runtime facilities.
///
/// It intentionally does not mutate `NOSV_CONFIG` or environment overrides: nOS-V
/// configuration is process-global and is consumed by native initialization. The
/// current builder is zero-sized but provides a stable place for future Rust-only
/// driver and instrumentation options.
#[cfg(feature = "io-uring")]
#[doc(hidden)]
pub type DefaultSubmissionEntry = crate::io_uring::raw::squeue::Entry;
#[cfg(not(feature = "io-uring"))]
#[doc(hidden)]
pub type DefaultSubmissionEntry = ();
#[cfg(feature = "io-uring")]
#[doc(hidden)]
pub type DefaultCompletionEntry = crate::io_uring::raw::cqueue::Entry;
#[cfg(not(feature = "io-uring"))]
#[doc(hidden)]
pub type DefaultCompletionEntry = ();

#[cfg(feature = "io-uring")]
#[doc(hidden)]
pub trait RuntimeSubmissionEntry:
    crate::io_uring::raw::squeue::EntryMarker + Send + 'static
{
}
#[cfg(feature = "io-uring")]
impl<T: crate::io_uring::raw::squeue::EntryMarker + Send + 'static> RuntimeSubmissionEntry for T {}
#[cfg(not(feature = "io-uring"))]
#[doc(hidden)]
pub trait RuntimeSubmissionEntry: Send + 'static {}
#[cfg(not(feature = "io-uring"))]
impl<T: Send + 'static> RuntimeSubmissionEntry for T {}

#[cfg(feature = "io-uring")]
#[doc(hidden)]
pub trait RuntimeCompletionEntry:
    crate::io_uring::raw::cqueue::EntryMarker + Send + 'static
{
}
#[cfg(feature = "io-uring")]
impl<T: crate::io_uring::raw::cqueue::EntryMarker + Send + 'static> RuntimeCompletionEntry for T {}
#[cfg(not(feature = "io-uring"))]
#[doc(hidden)]
pub trait RuntimeCompletionEntry: Send + 'static {}
#[cfg(not(feature = "io-uring"))]
impl<T: Send + 'static> RuntimeCompletionEntry for T {}

/// Configures Rust-layer runtime facilities without native side effects.
pub struct RuntimeBuilder<S = DefaultSubmissionEntry, C = DefaultCompletionEntry> {
    #[cfg(feature = "io-uring")]
    /// Configuration prepared before native runtime initialization.
    io_uring_config: crate::io_uring::IoUringConfig,
    /// Selects the SQE and CQE marker implementations when I/O is enabled.
    _io_entries: PhantomData<fn() -> (S, C)>,
    /// Prevents external struct literals.
    _private: (),
}

impl<S: RuntimeSubmissionEntry, C: RuntimeCompletionEntry> Default for RuntimeBuilder<S, C> {
    fn default() -> Self {
        Self {
            #[cfg(feature = "io-uring")]
            io_uring_config: crate::io_uring::IoUringConfig::default(),
            _io_entries: PhantomData,
            _private: (),
        }
    }
}
impl<S: RuntimeSubmissionEntry, C: RuntimeCompletionEntry> Clone for RuntimeBuilder<S, C> {
    fn clone(&self) -> Self {
        Self {
            #[cfg(feature = "io-uring")]
            io_uring_config: self.io_uring_config,
            _io_entries: PhantomData,
            _private: (),
        }
    }
}
impl<S: RuntimeSubmissionEntry, C: RuntimeCompletionEntry> std::fmt::Debug
    for RuntimeBuilder<S, C>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeBuilder").finish_non_exhaustive()
    }
}

impl RuntimeBuilder<DefaultSubmissionEntry, DefaultCompletionEntry> {
    /// Constructs a builder using the default ring entry widths.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<S: RuntimeSubmissionEntry, C: RuntimeCompletionEntry> RuntimeBuilder<S, C> {
    /// Selects the runtime-wide `io_uring` driver configuration.
    #[cfg(feature = "io-uring")]
    #[must_use]
    pub const fn io_uring_config(mut self, config: crate::io_uring::IoUringConfig) -> Self {
        self.io_uring_config = config;
        self
    }

    /// Initializes nOS-V, creates runtime task types, and publishes a runtime.
    ///
    /// Construction proceeds transactionally: the async type is created first,
    /// then optional driver types and tasks, and finally the shared core. Every
    /// failure path retires successfully created resources and balances native
    /// initialization before returning.
    ///
    /// # Errors
    ///
    /// Returns [`InitError::AlreadyInTask`] from any current nOS-V task context,
    /// or a structured configuration, kernel ring, or native resource error.
    pub fn build(&self) -> Result<Runtime<S, C>, InitError> {
        if ffi::current().is_some() {
            return Err(InitError::AlreadyInTask);
        }
        #[cfg(feature = "io-uring")]
        let prepared_io_uring = crate::io_uring::PreparedRing::<S, C>::new(self.io_uring_config)?;
        ffi::init()?;

        let task_type = match ffi::type_init(
            Some(task::run_callback),
            Some(task::completed_callback),
            Some(task::cost_callback),
            c"rust.async",
        ) {
            Ok(task_type) => task_type,
            Err(error) => {
                let _ = ffi::shutdown();
                return Err(InitError::Native(error));
            }
        };

        #[cfg(feature = "io-uring")]
        let io_uring_type = match ffi::type_init(
            Some(crate::io_uring::run_callback::<S, C>),
            Some(crate::io_uring::completed_callback::<S, C>),
            None,
            c"rust.io_uring",
        ) {
            Ok(io_uring_type) => io_uring_type,
            Err(error) => {
                let _ = ffi::type_destroy(task_type);
                let _ = ffi::shutdown();
                return Err(InitError::Native(error));
            }
        };
        #[cfg(feature = "io-uring")]
        let io_uring = match crate::io_uring::IoUringDriver::start(io_uring_type, prepared_io_uring)
        {
            Ok(driver) => driver,
            Err(error) => {
                let _ = ffi::type_destroy(io_uring_type);
                let _ = ffi::type_destroy(task_type);
                let _ = ffi::shutdown();
                return Err(InitError::Native(error));
            }
        };

        #[cfg(feature = "time")]
        let timer_type = match ffi::type_init(
            Some(crate::time::run_callback),
            Some(crate::time::completed_callback),
            None,
            c"rust.timer",
        ) {
            Ok(timer_type) => timer_type,
            Err(error) => {
                #[cfg(feature = "io-uring")]
                {
                    io_uring.shutdown_and_wait();
                    let _ = ffi::type_destroy(io_uring_type);
                }
                let _ = ffi::type_destroy(task_type);
                let _ = ffi::shutdown();
                return Err(InitError::Native(error));
            }
        };
        #[cfg(feature = "time")]
        let timer = match crate::time::TimerDriver::start(timer_type) {
            Ok(timer) => timer,
            Err(error) => {
                let _ = ffi::type_destroy(timer_type);
                #[cfg(feature = "io-uring")]
                {
                    io_uring.shutdown_and_wait();
                    let _ = ffi::type_destroy(io_uring_type);
                }
                let _ = ffi::type_destroy(task_type);
                let _ = ffi::shutdown();
                return Err(InitError::Native(error));
            }
        };

        // SAFETY: getpid has no preconditions.
        let pid = unsafe { libc::getpid() };
        let core = Arc::new(RuntimeCore {
            owner_thread: thread::current().id(),
            pid,
            task_type,
            #[cfg(feature = "time")]
            timer_type,
            #[cfg(feature = "time")]
            timer,
            state: Mutex::new(RuntimeState {
                lifecycle: Lifecycle::Running,
                tasks: HashMap::new(),
            }),
            drained: Condvar::new(),
            next_task_id: std::sync::atomic::AtomicU64::new(1),
        });
        Ok(Runtime {
            core,
            #[cfg(feature = "io-uring")]
            io_uring_type,
            #[cfg(feature = "io-uring")]
            io_uring,
            active: true,
            _entry_types: PhantomData,
            _not_send_or_sync: PhantomData,
        })
    }
}

/// An initialized nOS-V runtime owned by its creating thread.
///
/// `Runtime` is deliberately neither `Send` nor `Sync`: nOS-V requires each
/// pthread to balance its own init/shutdown calls. Use [`Runtime::handle`] to move
/// spawning capability to other threads. Dropping the runtime performs the same
/// cooperative, potentially unbounded drain as [`Runtime::shutdown`].
pub struct Runtime<S = DefaultSubmissionEntry, C = DefaultCompletionEntry>
where
    S: RuntimeSubmissionEntry,
    C: RuntimeCompletionEntry,
{
    /// Shared state used by handles, tasks, callbacks, and drivers.
    pub(crate) core: Arc<RuntimeCore>,
    #[cfg(feature = "io-uring")]
    /// Native descriptor type owned alongside the typed ring.
    io_uring_type: RawTaskType,
    #[cfg(feature = "io-uring")]
    /// Typed runtime-wide ring and driver.
    io_uring: Arc<crate::io_uring::IoUringDriver<S, C>>,
    /// Whether this value still owes the native runtime a shutdown operation.
    active: bool,
    /// Keeps entry type selection present when the I/O feature is disabled.
    _entry_types: PhantomData<fn() -> (S, C)>,
    /// `Rc` marker that statically pins lifecycle ownership to one pthread.
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl Runtime<DefaultSubmissionEntry, DefaultCompletionEntry> {
    /// Initializes a runtime with default ring entry widths.
    pub fn new() -> Result<Self, InitError> {
        RuntimeBuilder::new().build()
    }

    /// Returns a builder using the default ring entry widths.
    pub fn builder() -> RuntimeBuilder<DefaultSubmissionEntry, DefaultCompletionEntry> {
        RuntimeBuilder::new()
    }
}

impl<S: RuntimeSubmissionEntry, C: RuntimeCompletionEntry> Runtime<S, C> {
    /// Clones a thread-safe spawning and query handle.
    ///
    /// The handle may cross threads, but every operation checks the runtime
    /// lifecycle and process generation before touching native state. Keeping a
    /// handle alive does not keep `Runtime::shutdown` from closing the runtime.
    pub fn handle(&self) -> Handle {
        Handle {
            core: self.core.clone(),
        }
    }

    /// Clones the thread-safe capability for this runtime's typed ring.
    #[cfg(feature = "io-uring")]
    pub fn io_uring_handle(&self) -> crate::io_uring::IoUringHandle<S, C> {
        crate::io_uring::IoUringHandle::new(self.io_uring.clone(), &self.core)
    }

    /// Drives a possibly borrowed, non-`Send` root future on the owner pthread.
    ///
    /// The method attaches that pthread as an external nOS-V task, polls the root
    /// directly, and calls `nosv_pause` after `Poll::Pending`. Its parker waker
    /// submits the attached descriptor; nOS-V's early-wake counter prevents a wake
    /// between polling and pausing from being lost. The attached stack remains on
    /// the caller, which is why `F` need not be `Send` or `'static`.
    ///
    /// The parker and current-runtime TLS are closed before detach. A panic from
    /// the root future is caught only long enough to perform that cleanup and is
    /// then resumed on the caller.
    ///
    /// # Errors
    ///
    /// Rejects calls from a non-owner thread, a forked child, a closing runtime,
    /// another nOS-V task, or a nested `block_on`. Native attach, pause, and detach
    /// failures are returned as [`BlockOnError::Native`].
    pub fn try_block_on<F: Future>(&self, future: F) -> Result<F::Output, BlockOnError> {
        if thread::current().id() != self.core.owner_thread {
            return Err(BlockOnError::WrongThread);
        }
        if !self.core.process_matches() {
            return Err(BlockOnError::ForkedProcess);
        }
        if self.core.lock_state().lifecycle != Lifecycle::Running {
            return Err(BlockOnError::RuntimeClosed);
        }
        if ffi::current().is_some() {
            return Err(BlockOnError::AlreadyInTask);
        }
        if IN_BLOCK_ON.with(|flag| flag.replace(true)) {
            return Err(BlockOnError::Nested);
        }

        let raw = match ffi::attach(c"rust.block_on") {
            Ok(raw) => raw,
            Err(error) => {
                IN_BLOCK_ON.with(|flag| flag.set(false));
                return Err(BlockOnError::Native(error));
            }
        };
        let parker = Arc::new(Parker {
            gate: Mutex::new(ParkerGate {
                raw,
                pid: self.core.pid,
                live: true,
                wake_submitted: true,
            }),
        });
        let waker = Waker::from(parker.clone());
        let mut future = pin!(future);
        let handle = self.handle();
        let entered = enter_current(handle);

        let driven =
            panic::catch_unwind(AssertUnwindSafe(|| -> Result<F::Output, BlockOnError> {
                loop {
                    parker.begin_poll();
                    let mut context = Context::from_waker(&waker);
                    match Future::poll(future.as_mut(), &mut context) {
                        Poll::Ready(output) => return Ok(output),
                        Poll::Pending => ffi::pause().map_err(BlockOnError::Native)?,
                    }
                }
            }));

        parker.close();
        drop(entered);
        IN_BLOCK_ON.with(|flag| flag.set(false));
        let detached = ffi::detach().map_err(BlockOnError::Native);
        match driven {
            Ok(Ok(output)) => {
                detached?;
                Ok(output)
            }
            Ok(Err(error)) => {
                let _ = detached;
                Err(error)
            }
            Err(payload) => {
                let _ = detached;
                panic::resume_unwind(payload)
            }
        }
    }

    /// Drives a root future and panics if attached-thread setup fails.
    ///
    /// This convenience form delegates to [`Runtime::try_block_on`]. It does not
    /// convert a panic from the root future; that original panic is resumed after
    /// native detach just as it is in the fallible form.
    pub fn block_on<F: Future>(&self, future: F) -> F::Output {
        self.try_block_on(future)
            .unwrap_or_else(|error| panic!("nOS-V block_on failed: {error}"))
    }

    /// Closes spawning, cooperatively drains tasks, and shuts nOS-V down.
    ///
    /// Every registered future receives an abort request. Shutdown then waits for
    /// each callback to observe cancellation, drop user state, destroy its native
    /// descriptor, and leave the registry. Drivers are kept alive during that
    /// process and retired before the final native shutdown call.
    ///
    /// There is intentionally no timeout: finalizing nOS-V while a descriptor is
    /// still reachable would be unsafe. A future that never returns from `poll`
    /// can therefore make this method wait indefinitely.
    pub fn shutdown(mut self) -> Result<(), ShutdownError> {
        let result = self.shutdown_inner();
        if result.is_ok() {
            self.active = false;
        }
        result
    }

    /// Implements idempotent shutdown for both explicit shutdown and `Drop`.
    ///
    /// The lifecycle transition and registry snapshot occur under one mutex, then
    /// abort requests are made without holding it. The condition variable releases
    /// the lock while callbacks finish, avoiding a completion/shutdown deadlock.
    fn shutdown_inner(&self) -> Result<(), ShutdownError> {
        if thread::current().id() != self.core.owner_thread {
            return Err(ShutdownError::WrongThread);
        }
        if !self.core.process_matches() {
            return Err(ShutdownError::ForkedProcess);
        }

        let tasks = {
            let mut state = self.core.lock_state();
            match state.lifecycle {
                Lifecycle::Closed => return Ok(()),
                Lifecycle::Running => state.lifecycle = Lifecycle::Closing,
                Lifecycle::Closing => {}
            }
            state.tasks.values().cloned().collect::<Vec<_>>()
        };
        for task in tasks {
            task.request_abort();
        }

        let mut state = self.core.lock_state();
        while !state.tasks.is_empty() {
            state = self
                .core
                .drained
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        drop(state);

        #[cfg(feature = "io-uring")]
        {
            self.io_uring.shutdown_and_wait();
            ffi::type_destroy(self.io_uring_type).map_err(ShutdownError::Native)?;
        }

        #[cfg(feature = "time")]
        {
            self.core.timer.shutdown_and_wait();
            ffi::type_destroy(self.core.timer_type).map_err(ShutdownError::Native)?;
        }
        ffi::type_destroy(self.core.task_type).map_err(ShutdownError::Native)?;
        ffi::shutdown().map_err(ShutdownError::Native)?;
        self.core.lock_state().lifecycle = Lifecycle::Closed;
        Ok(())
    }
}

impl<S: RuntimeSubmissionEntry, C: RuntimeCompletionEntry> Drop for Runtime<S, C> {
    /// Attempts the same cooperative shutdown when explicit shutdown was omitted.
    ///
    /// `Drop` cannot report an error, so it logs a teardown failure. It never moves
    /// shutdown to another thread or guesses that live native state may be freed.
    fn drop(&mut self) {
        if self.active {
            if let Err(error) = self.shutdown_inner() {
                eprintln!("nOS-V runtime drop could not shut down safely: {error}");
            } else {
                self.active = false;
            }
        }
    }
}

/// A clonable, thread-safe capability for a runtime generation.
///
/// `Handle` is `Send + Sync` and is the intended way to spawn from foreign
/// pthreads. It does not expose native descriptors, cannot perform shutdown, and
/// becomes inert after close or `fork` even if clones remain alive.
#[derive(Clone)]
pub struct Handle {
    /// Shared runtime generation; all operations validate it before FFI.
    pub(crate) core: Arc<RuntimeCore>,
}

impl Handle {
    /// Returns the runtime handle installed for the currently polled future.
    ///
    /// Spawned-task callbacks and `block_on` install this TLS value with an unwind-
    /// safe guard. Calls made outside either polling scope return [`RuntimeClosed`]
    /// and perform no native operation.
    pub fn try_current() -> Result<Self, RuntimeClosed> {
        CURRENT
            .with(|current| current.borrow().clone())
            .ok_or(RuntimeClosed)
    }

    /// Creates and submits a `Send + 'static` future on this runtime.
    ///
    /// Construction allocates Rust ownership first, creates a non-parallel and
    /// non-joinable native descriptor, stores one unaligned owner pointer in its
    /// metadata, registers the task, and submits while holding its descriptor gate.
    /// The `Send` bound permits nOS-V to resume polling on another worker pthread.
    ///
    /// Dropping the returned handle detaches rather than cancels the task.
    pub fn spawn<F, T>(&self, future: F) -> Result<JoinHandle<T>, SpawnError>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        task::spawn_on(self, task::TaskConfig::default(), future)
    }

    /// Starts configuring attributes fixed before native submission.
    ///
    /// Returning a borrowing builder makes the runtime association explicit and
    /// prevents priority or affinity mutation after spawning.
    pub fn task(&self) -> TaskBuilder<'_> {
        TaskBuilder::new(self)
    }

    /// Returns a topology capability bound to this runtime generation.
    ///
    /// The capability stores only a weak core reference. Every query reacquires
    /// the lifecycle lock through its FFI call, so retaining it cannot extend the
    /// runtime or race native shutdown.
    pub fn topology(&self) -> Result<Topology, RuntimeClosed> {
        self.core.ensure_running()?;
        Ok(Topology {
            runtime: Arc::downgrade(&self.core),
        })
    }

    /// Takes a validated snapshot of nOS-V shared-memory statistics.
    ///
    /// The runtime state lock is held through all three native queries, making the
    /// snapshot linearizable with shutdown. Returned sizes and pressure are checked
    /// before becoming safe Rust values.
    pub fn memory_stats(&self) -> Result<MemoryStats, NativeError> {
        if !self.core.process_matches() {
            return Err(NativeError::NotInitialized);
        }
        let state = self.core.lock_state();
        if state.lifecycle != Lifecycle::Running {
            return Err(NativeError::NotInitialized);
        }
        MemoryStats::query()
    }
}

/// Restores the previous current-runtime TLS value on scope exit.
///
/// Polling scopes may nest internally, so the guard records rather than simply
/// clears the previous handle. Its `Drop` implementation also runs during unwind.
struct CurrentGuard {
    /// Handle that occupied TLS before the current polling scope.
    previous: Option<Handle>,
}
impl Drop for CurrentGuard {
    /// Reinstalls the previous TLS value without invoking native code.
    fn drop(&mut self) {
        CURRENT.with(|current| {
            current.replace(self.previous.take());
        });
    }
}

/// Installs `handle` as current for the lifetime of the returned guard.
///
/// Callers bind the anonymous guard to their poll scope. Returning `impl Drop`
/// prevents code outside this module from inspecting or forgetting restoration
/// details accidentally.
pub(crate) fn enter_current(handle: Handle) -> impl Drop {
    let previous = CURRENT.with(|current| current.replace(Some(handle)));
    CurrentGuard { previous }
}

/// Thread-safe waker target for the attached `block_on` task.
///
/// Cloned standard wakers hold this object instead of a borrowed pointer. Its gate
/// serializes native submit with close and makes late wakes harmless.
struct Parker {
    /// Authoritative attached-descriptor and wake-coalescing state.
    gate: Mutex<ParkerGate>,
}
/// Mutable state protected by an attached parker's mutex.
struct ParkerGate {
    /// Implicit external task returned by `nosv_attach`.
    raw: ffi::RawTask,
    /// Process generation used to reject fork-inherited wakers.
    pid: libc::pid_t,
    /// Whether wake is still allowed to submit the attached descriptor.
    live: bool,
    /// Whether this poll epoch already has a native wake in flight.
    wake_submitted: bool,
}

impl Parker {
    /// Locks parker state and recovers it after a caught panic.
    fn lock(&self) -> MutexGuard<'_, ParkerGate> {
        lock(&self.gate)
    }
    /// Opens a new poll epoch so its first wake may submit exactly once.
    fn begin_poll(&self) {
        self.lock().wake_submitted = false;
    }
    /// Permanently makes late root-future wakers no-ops before detach.
    fn close(&self) {
        self.lock().live = false;
    }
}

impl Wake for Parker {
    /// Schedules the attached task, consuming this waker reference.
    ///
    /// Delegating to `wake_by_ref` keeps both standard `Wake` entry points on the
    /// same gated submit protocol.
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    /// Coalesces a root wake and submits the attached descriptor if still live.
    ///
    /// The gate is held through `nosv_submit`, closing check-versus-detach races. A
    /// valid ordinary submit failure is an internal invariant violation, so the
    /// process aborts rather than risking continued use of uncertain native state.
    fn wake_by_ref(self: &Arc<Self>) {
        let mut gate = self.lock();
        // SAFETY: getpid has no preconditions. An inherited parker must not
        // access the native state reset by nOS-V's at-fork handler.
        if gate.pid != unsafe { libc::getpid() } {
            return;
        }
        if !gate.live || gate.wake_submitted {
            return;
        }
        gate.wake_submitted = true;
        if let Err(error) = ffi::submit(gate.raw) {
            eprintln!("nOS-V rejected an attached-thread wake: {error}");
            std::process::abort();
        }
    }
}

/// Upgrades a task's weak runtime reference into a normal `Handle`.
///
/// Returning `None` lets a callback terminate safely if the runtime core has
/// already disappeared; no raw native pointer is exposed in either outcome.
pub(crate) fn weak_handle(core: &Weak<RuntimeCore>) -> Option<Handle> {
    core.upgrade().map(|core| Handle { core })
}
