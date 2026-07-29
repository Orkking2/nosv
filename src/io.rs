//! Completion-native owned I/O primitives.
//!
//! Operations submitted through this module retain every buffer and descriptor
//! until the kernel reports the original operation terminal. Dropping a future
//! requests cancellation, but a kernel side effect can still win that race.

use crate::{RuntimeClosed, runtime::RuntimeCore, util::lock};
use std::{
    future::Future,
    io,
    pin::Pin,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Waker},
};
use uring::squeue;

/// A result paired with the owned buffer supplied to an operation.
pub type BufResult<T, B> = (io::Result<T>, B);

/// A stable initialized buffer that may be retained by the kernel.
///
/// # Safety
///
/// Implementations must return the same allocation and initialized byte count
/// while ownership is held by an I/O operation. The pointer must remain valid
/// for reads of `bytes_init()` bytes when the value is moved.
pub unsafe trait IoBuf: Send + 'static {
    /// Returns the stable start of the allocation.
    fn stable_ptr(&self) -> *const u8;

    /// Returns the number of initialized bytes available for writing.
    fn bytes_init(&self) -> usize;
}

/// A stable buffer whose allocation may be initialized by the kernel.
///
/// # Safety
///
/// In addition to [`IoBuf`]'s requirements, `stable_mut_ptr()` must permit writes
/// of `bytes_total()` bytes. `set_init(n)` must make the first `n` bytes safe to
/// read, and is called only after the driver validates a completion length.
pub unsafe trait IoBufMut: IoBuf {
    /// Returns the stable mutable start of the allocation.
    fn stable_mut_ptr(&mut self) -> *mut u8;

    /// Returns the allocation size writable by the kernel.
    fn bytes_total(&self) -> usize;

    /// Marks the first `n` bytes initialized.
    ///
    /// # Safety
    ///
    /// The caller must have initialized the first `n` bytes and `n` must not
    /// exceed [`IoBufMut::bytes_total`].
    unsafe fn set_init(&mut self, n: usize);
}

// SAFETY: Vec keeps its allocation stable while exclusively owned, and its
// initialized prefix is exactly its length.
unsafe impl IoBuf for Vec<u8> {
    fn stable_ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    fn bytes_init(&self) -> usize {
        self.len()
    }
}

// SAFETY: Vec exposes its full capacity for writes and set_len publishes only a
// prefix that the operation has reported initialized.
unsafe impl IoBufMut for Vec<u8> {
    fn stable_mut_ptr(&mut self) -> *mut u8 {
        self.as_mut_ptr()
    }

    fn bytes_total(&self) -> usize {
        self.capacity()
    }

    unsafe fn set_init(&mut self, n: usize) {
        // SAFETY: required by the trait method's caller contract.
        unsafe { self.set_len(n) };
    }
}

// SAFETY: a boxed slice has a stable, fully initialized allocation.
unsafe impl IoBuf for Box<[u8]> {
    fn stable_ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    fn bytes_init(&self) -> usize {
        self.len()
    }
}

// SAFETY: every byte of a boxed byte slice is writable and initialized.
unsafe impl IoBufMut for Box<[u8]> {
    fn stable_mut_ptr(&mut self) -> *mut u8 {
        self.as_mut_ptr()
    }

    fn bytes_total(&self) -> usize {
        self.len()
    }

    unsafe fn set_init(&mut self, n: usize) {
        debug_assert!(n <= self.len());
    }
}

// SAFETY: a static slice remains valid for the program lifetime.
unsafe impl IoBuf for &'static [u8] {
    fn stable_ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    fn bytes_init(&self) -> usize {
        self.len()
    }
}

// SAFETY: exclusive ownership of a static mutable slice provides a stable,
// fully initialized writable allocation.
unsafe impl IoBuf for &'static mut [u8] {
    fn stable_ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    fn bytes_init(&self) -> usize {
        self.len()
    }
}

// SAFETY: the whole static mutable slice is writable and initialized.
unsafe impl IoBufMut for &'static mut [u8] {
    fn stable_mut_ptr(&mut self) -> *mut u8 {
        self.as_mut_ptr()
    }

    fn bytes_total(&self) -> usize {
        self.len()
    }

    unsafe fn set_init(&mut self, n: usize) {
        debug_assert!(n <= self.len());
    }
}

/// Non-generic capability for completion-native operations on one runtime.
#[derive(Clone)]
pub struct IoHandle {
    /// Erased driver shared by all entry-width configurations.
    pub(crate) driver: Arc<dyn ErasedOwnedDriver>,
    /// Runtime generation used for lifecycle and fork checks.
    pub(crate) core: Weak<RuntimeCore>,
}

impl std::fmt::Debug for IoHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IoHandle").finish_non_exhaustive()
    }
}

impl IoHandle {
    /// Returns the owned-I/O capability installed for the currently polled task.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeClosed`] outside a runtime polling scope, after shutdown,
    /// or in a process created by `fork` from the runtime process.
    pub fn try_current() -> Result<Self, RuntimeClosed> {
        crate::runtime::Handle::try_current()?.io_handle()
    }

    /// Checks that this handle still belongs to a running runtime generation.
    pub(crate) fn ensure_running(&self) -> Result<(), RuntimeClosed> {
        self.core.upgrade().ok_or(RuntimeClosed)?.ensure_running()
    }
}

/// Cancellation and wake state shared by a future and its driver record.
pub(crate) struct OwnedControl {
    /// Set by future drop before notifying the driver.
    cancelled: AtomicBool,
    /// Published after the typed result has been stored.
    completed: AtomicBool,
    /// Caller waiting for completion.
    waker: Mutex<Option<Waker>>,
}

impl OwnedControl {
    /// Allocates an idle control block.
    fn new() -> Arc<Self> {
        Arc::new(Self {
            cancelled: AtomicBool::new(false),
            completed: AtomicBool::new(false),
            waker: Mutex::new(None),
        })
    }

    /// Reports whether caller drop requested cancellation.
    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Marks cancellation before the driver notification linearization step.
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Publishes result availability and returns the registered waiter.
    fn publish(&self) -> Option<Waker> {
        self.completed.store(true, Ordering::Release);
        lock(&self.waker).take()
    }
}

/// Type-erased work record transferred through the runtime-wide queue.
pub(crate) trait OwnedHandler: Send {
    /// Opcode used for retained probe validation.
    fn opcode(&self) -> u8;
    /// Builds an SQE while all pointer-bearing fields are stable in this box.
    fn entry(&mut self) -> squeue::Entry;
    /// Converts a terminal CQE into the public typed result.
    fn complete(self: Box<Self>, result: i32) -> Option<Waker>;
    /// Completes without submission when admission or capability validation fails.
    fn fail(self: Box<Self>, error: io::Error) -> Option<Waker>;
}

/// Work-specific SQE construction and result conversion.
pub(crate) trait OpWork<T>: Send + 'static {
    /// Opcode used by this work item.
    fn opcode(&self) -> u8;
    /// Builds the SQE after the work item is pointer-stable.
    fn entry(&mut self) -> squeue::Entry;
    /// Handles a terminal kernel result.
    fn complete(self: Box<Self>, result: i32) -> T;
    /// Handles a pre-submission failure.
    fn fail(self: Box<Self>, error: io::Error) -> T;
}

/// Typed result slot shared by a handler and its future.
struct OwnedState<T> {
    /// Erased lifecycle state visible to the driver.
    control: Arc<OwnedControl>,
    /// Exactly one terminal result.
    result: Mutex<Option<T>>,
}

/// Erases a work/result pair for the driver queue.
struct TypedHandler<T> {
    /// Result destination.
    state: Arc<OwnedState<T>>,
    /// Resources and operation-specific behavior.
    work: Box<dyn OpWork<T>>,
}

impl<T: Send + 'static> OwnedHandler for TypedHandler<T> {
    fn opcode(&self) -> u8 {
        self.work.opcode()
    }

    fn entry(&mut self) -> squeue::Entry {
        self.work.entry()
    }

    fn complete(self: Box<Self>, result: i32) -> Option<Waker> {
        let value = self.work.complete(result);
        *lock(&self.state.result) = Some(value);
        self.state.control.publish()
    }

    fn fail(self: Box<Self>, error: io::Error) -> Option<Waker> {
        let value = self.work.fail(error);
        *lock(&self.state.result) = Some(value);
        self.state.control.publish()
    }
}

/// One record in the lock-free producer queue.
pub(crate) struct QueuedOwnedOp {
    /// Cancellation identity and future waiter.
    pub(crate) control: Arc<OwnedControl>,
    /// Resources retained through terminal completion.
    pub(crate) handler: Box<dyn OwnedHandler>,
}

// SAFETY: UBQ transfers each record by value to exactly one consumer. No shared
// reference to the handler is exposed while it occupies a queue slot, so Send is
// the actual resource requirement even though UBQ conservatively requires Sync.
unsafe impl Sync for QueuedOwnedOp {}

/// Entry-width-erased operations implemented by the typed ring driver.
pub(crate) trait ErasedOwnedDriver: Send + Sync {
    /// Publishes one lazy operation, returning it unchanged after closure.
    fn enqueue(&self, operation: QueuedOwnedOp) -> Result<(), QueuedOwnedOp>;
    /// Publishes an exact batch as one gap-free FIFO reservation.
    #[allow(dead_code)]
    fn enqueue_batch(&self, operations: Vec<QueuedOwnedOp>) -> Result<(), Vec<QueuedOwnedOp>>;
    /// Notifies and, when already staged, targets a cancelled operation.
    fn cancel(&self, control: &Arc<OwnedControl>);
}

/// Lazy, cancellation-safe, single-completion operation future.
pub(crate) struct OwnedOp<T> {
    /// Explicit driver/runtime capability.
    handle: IoHandle,
    /// Shared typed result state.
    state: Arc<OwnedState<T>>,
    /// Handler retained locally until first poll.
    handler: Option<Box<dyn OwnedHandler>>,
    /// Whether ownership was transferred to the driver.
    queued: bool,
    /// Prevents polling after the output was taken.
    done: bool,
}

impl<T: Send + 'static> OwnedOp<T> {
    /// Creates a lazy operation without touching the driver.
    pub(crate) fn new<W: OpWork<T>>(handle: &IoHandle, work: W) -> Self {
        let control = OwnedControl::new();
        let state = Arc::new(OwnedState {
            control: control.clone(),
            result: Mutex::new(None),
        });
        let handler = Box::new(TypedHandler {
            state: state.clone(),
            work: Box::new(work),
        });
        Self {
            handle: handle.clone(),
            state,
            handler: Some(handler),
            queued: false,
            done: false,
        }
    }
}

impl<T: Send + 'static> Future for OwnedOp<T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        if self.done {
            panic!("owned I/O operation polled after completion");
        }

        if !self.queued {
            let handler = self.handler.take().expect("unsubmitted owned handler");
            if let Err(_closed) = self.handle.ensure_running() {
                if let Some(waker) = handler.fail(runtime_closed_error()) {
                    waker.wake();
                }
            } else {
                let operation = QueuedOwnedOp {
                    control: self.state.control.clone(),
                    handler,
                };
                match self.handle.driver.enqueue(operation) {
                    Ok(()) => self.queued = true,
                    Err(operation) => {
                        if let Some(waker) = operation.handler.fail(runtime_closed_error()) {
                            waker.wake();
                        }
                    }
                }
            }
        }

        if self.state.control.completed.load(Ordering::Acquire) {
            self.done = true;
            return Poll::Ready(
                lock(&self.state.result)
                    .take()
                    .expect("published owned result missing"),
            );
        }

        {
            let mut slot = lock(&self.state.control.waker);
            if slot.as_ref().is_none_or(|old| !old.will_wake(cx.waker())) {
                *slot = Some(cx.waker().clone());
            }
        }
        if self.state.control.completed.load(Ordering::Acquire) {
            self.done = true;
            Poll::Ready(
                lock(&self.state.result)
                    .take()
                    .expect("published owned result missing"),
            )
        } else {
            Poll::Pending
        }
    }
}

impl<T> Drop for OwnedOp<T> {
    fn drop(&mut self) {
        if self.queued && !self.done {
            self.state.control.cancel();
            self.handle.driver.cancel(&self.state.control);
        }
    }
}

/// Converts a negative CQE result into an ordinary I/O error.
pub(crate) fn result(result: i32) -> io::Result<usize> {
    if result < 0 {
        Err(io::Error::from_raw_os_error(-result))
    } else {
        Ok(result as usize)
    }
}

/// Creates the uniform error returned after runtime closure.
pub(crate) fn runtime_closed_error() -> io::Error {
    io::Error::new(io::ErrorKind::BrokenPipe, "nOS-V I/O runtime is closed")
}

/// Creates an unsupported-opcode error.
pub(crate) fn unsupported_opcode_error() -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        "kernel io_uring does not support this operation",
    )
}

#[cfg(feature = "tokio-compat")]
pub use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
