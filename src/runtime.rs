//! Runtime construction, lifecycle, handles, and attached-thread `block_on`.

use crate::{
    error::{BlockOnError, InitError, NativeError, RuntimeClosed, ShutdownError, SpawnError},
    ffi::{self, RawTaskType},
    memory::MemoryStats,
    task::{self, JoinHandle, TaskBuilder, TaskCore, TaskId},
    topology::Topology,
};
use std::{
    cell::{Cell, RefCell}, collections::HashMap, future::Future, marker::PhantomData, panic::{self, AssertUnwindSafe}, pin::pin, rc::Rc, sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak, atomic::{AtomicU64, Ordering::Relaxed}}, task::{Context, Poll, Wake, Waker}, thread::{self, ThreadId},
};

thread_local! {
    static CURRENT: RefCell<Option<Handle>> = const { RefCell::new(None) };
    static IN_BLOCK_ON: Cell<bool> = const { Cell::new(false) };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Lifecycle {
    Running,
    Closing,
    Closed,
}

pub(crate) struct RuntimeState {
    pub(crate) lifecycle: Lifecycle,
    pub(crate) tasks: HashMap<TaskId, Arc<TaskCore>>,
}

pub(crate) struct RuntimeCore {
    pub(crate) owner_thread: ThreadId,
    pub(crate) pid: libc::pid_t,
    pub(crate) task_type: RawTaskType,
    #[cfg(feature = "time")]
    pub(crate) timer_type: RawTaskType,
    #[cfg(feature = "time")]
    pub(crate) timer: Arc<crate::time::TimerDriver>,
    pub(crate) state: Mutex<RuntimeState>,
    pub(crate) drained: Condvar,
    next_task_id: AtomicU64,
}

impl RuntimeCore {
    pub(crate) fn lock_state(&self) -> MutexGuard<'_, RuntimeState> {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn process_matches(&self) -> bool {
        // SAFETY: getpid has no preconditions.
        self.pid == unsafe { libc::getpid() }
    }

    pub(crate) fn next_task_id(&self) -> TaskId {
        let id = self
            .next_task_id
            .fetch_add(1, Relaxed);
        if id == u64::MAX {
            eprintln!("nOS-V Rust runtime exhausted its task identifier space");
            std::process::abort();
        }
        TaskId(id)
    }

    pub(crate) fn task_completed(&self, id: TaskId) {
        let mut state = self.lock_state();
        state.tasks.remove(&id);
        if state.tasks.is_empty() {
            self.drained.notify_all();
        }
    }

    pub(crate) fn ensure_running(&self) -> Result<(), RuntimeClosed> {
        if !self.process_matches() || self.lock_state().lifecycle != Lifecycle::Running {
            Err(RuntimeClosed)
        } else {
            Ok(())
        }
    }
}

/// Builder for Rust-layer runtime facilities.
#[derive(Clone, Debug, Default)]
pub struct RuntimeBuilder {
    _private: (),
}

impl RuntimeBuilder {
    /// Constructs a builder with safe defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Initializes nOS-V and constructs the runtime.
    pub fn build(&self) -> Result<Runtime, InitError> {
        if ffi::current().is_some() {
            return Err(InitError::AlreadyInTask);
        }
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

        #[cfg(feature = "time")]
        let timer_type = match ffi::type_init(
            Some(crate::time::run_callback),
            Some(crate::time::completed_callback),
            None,
            c"rust.timer",
        ) {
            Ok(timer_type) => timer_type,
            Err(error) => {
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
            active: true,
            _not_send_or_sync: PhantomData,
        })
    }
}

/// An initialized nOS-V runtime owned by its creating thread.
pub struct Runtime {
    pub(crate) core: Arc<RuntimeCore>,
    active: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl Runtime {
    /// Initializes a runtime with default Rust-layer facilities.
    pub fn new() -> Result<Self, InitError> {
        RuntimeBuilder::new().build()
    }
    /// Returns a runtime builder.
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::new()
    }
    /// Returns a thread-safe handle.
    pub fn handle(&self) -> Handle {
        Handle {
            core: self.core.clone(),
        }
    }

    /// Drives a possibly borrowed, non-`Send` root future on the owner thread.
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

    /// Drives a root future, panicking if runtime setup fails.
    pub fn block_on<F: Future>(&self, future: F) -> F::Output {
        self.try_block_on(future)
            .unwrap_or_else(|error| panic!("nOS-V block_on failed: {error}"))
    }

    /// Cooperatively drains tasks and shuts nOS-V down on the owner thread.
    pub fn shutdown(mut self) -> Result<(), ShutdownError> {
        let result = self.shutdown_inner();
        if result.is_ok() {
            self.active = false;
        }
        result
    }

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

impl Drop for Runtime {
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

/// A clonable, thread-safe handle to a live runtime.
#[derive(Clone)]
pub struct Handle {
    pub(crate) core: Arc<RuntimeCore>,
}

impl Handle {
    /// Returns the handle installed for the currently polled future.
    pub fn try_current() -> Result<Self, RuntimeClosed> {
        CURRENT
            .with(|current| current.borrow().clone())
            .ok_or(RuntimeClosed)
    }

    /// Spawns a `Send + 'static` future.
    pub fn spawn<F, T>(&self, future: F) -> Result<JoinHandle<T>, SpawnError>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        task::spawn_on(self, task::TaskConfig::default(), future)
    }

    /// Starts configuring a task before submission.
    pub fn task(&self) -> TaskBuilder<'_> {
        TaskBuilder::new(self)
    }

    /// Returns a safe topology query capability.
    pub fn topology(&self) -> Result<Topology, RuntimeClosed> {
        self.core.ensure_running()?;
        Ok(Topology {
            runtime: Arc::downgrade(&self.core),
        })
    }

    /// Takes a checked shared-memory snapshot.
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

struct CurrentGuard {
    previous: Option<Handle>,
}
impl Drop for CurrentGuard {
    fn drop(&mut self) {
        CURRENT.with(|current| {
            current.replace(self.previous.take());
        });
    }
}

pub(crate) fn enter_current(handle: Handle) -> impl Drop {
    let previous = CURRENT.with(|current| current.replace(Some(handle)));
    CurrentGuard { previous }
}

struct Parker {
    gate: Mutex<ParkerGate>,
}
struct ParkerGate {
    raw: ffi::RawTask,
    pid: libc::pid_t,
    live: bool,
    wake_submitted: bool,
}

impl Parker {
    fn lock(&self) -> MutexGuard<'_, ParkerGate> {
        self.gate.lock().unwrap_or_else(|p| p.into_inner())
    }
    fn begin_poll(&self) {
        self.lock().wake_submitted = false;
    }
    fn close(&self) {
        self.lock().live = false;
    }
}

impl Wake for Parker {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        let mut gate = self.lock();
        // SAFETY: getpid has no preconditions. An inherited parker must not
        // access the native state reset by nOS-V.s at-fork handler.
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

pub(crate) fn weak_handle(core: &Weak<RuntimeCore>) -> Option<Handle> {
    core.upgrade().map(|core| Handle { core })
}
