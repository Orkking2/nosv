//! Completion-native I/O handles and owned-buffer contracts.
//!
//! The safe [`crate::fs`] and [`crate::net`] operations move their buffers, paths,
//! addresses, and descriptor references into the runtime-wide I/O driver. Those
//! resources remain alive until the original kernel operation reaches a terminal
//! completion, even when the caller drops its future. Every buffer-taking method
//! returns the buffer together with its operational result.
//!
//! Dropping a submitted future requests cancellation and prevents later delivery
//! to that future. Cancellation does not roll back a kernel operation that already
//! won the race: a file write, connection attempt, or other side effect may still
//! occur. The optional `tokio-compat` feature uses readiness plus nonblocking
//! syscalls for borrowed TCP I/O, so a poll that returns `Pending` performs no
//! borrowed-buffer I/O.

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

/// An operation result paired with the original owned buffer.
///
/// The buffer is returned on success, kernel errors, validation errors, unsupported
/// opcodes, and runtime closure. For mutable operations, successfully initialized
/// bytes remain reflected in the returned buffer even when a later exact/all loop
/// fails.
pub type BufResult<T, B> = (io::Result<T>, B);

/// An owned buffer with a stable initialized byte prefix.
///
/// Implementations supplied by this crate cover [`Vec<u8>`], [`Box<[u8]>`],
/// [`&'static [u8]`], and [`&'static mut [u8]`]. File and socket write
/// operations read exactly the first [`bytes_init`](Self::bytes_init) bytes.
///
/// # Safety
///
/// Moving the implementing value must not change the allocation address returned
/// by [`stable_ptr`](Self::stable_ptr). That pointer must remain valid for reads of
/// `bytes_init()` bytes, and the first `bytes_init()` bytes must remain initialized,
/// for the entire time the operation owns the value. `bytes_init()` must not exceed
/// the allocation and must remain consistent with the pointer.
pub unsafe trait IoBuf: Send + 'static {
    /// Returns the stable start of the allocation.
    ///
    /// The pointer may be dangling for a zero-length allocation, but it must satisfy
    /// [`IoBuf`]'s validity requirements whenever [`bytes_init`](Self::bytes_init)
    /// is nonzero.
    fn stable_ptr(&self) -> *const u8;

    /// Returns the initialized prefix length readable by an I/O operation.
    fn bytes_init(&self) -> usize;
}

/// An [`IoBuf`] whose allocation may be initialized by an I/O operation.
///
/// Reads and receives begin at the allocation start and may write up to
/// [`bytes_total`](Self::bytes_total) bytes. For a [`Vec<u8>`], this is its capacity,
/// not its current length; existing initialized bytes may be overwritten. Boxed and
/// static mutable slices expose their entire length.
///
/// # Safety
///
/// In addition to [`IoBuf`]'s requirements,
/// [`stable_mut_ptr`](Self::stable_mut_ptr) must remain valid for writes of
/// `bytes_total()` bytes while the operation owns the value. `bytes_total()` must
/// not exceed the allocation. [`set_init`](Self::set_init) must soundly publish the
/// initialized prefix reported by a validated kernel completion.
pub unsafe trait IoBufMut: IoBuf {
    /// Returns the stable mutable start of the writable allocation.
    ///
    /// As with [`IoBuf::stable_ptr`], a zero-sized allocation need not point to
    /// writable storage.
    fn stable_mut_ptr(&mut self) -> *mut u8;

    /// Returns the total number of bytes writable from `stable_mut_ptr()`.
    fn bytes_total(&self) -> usize;

    /// Marks at least the first `n` allocation bytes initialized.
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

/// A cloneable, non-generic capability for one runtime's owned I/O driver.
///
/// Unlike [`crate::io_uring::IoUringHandle`], this handle erases the runtime's SQE
/// and CQE widths and is therefore suitable for [`crate::fs::File`] and
/// [`crate::net`] types. Clones are `Send + Sync`; keeping a clone alive does not
/// keep the runtime open. Operations validate the runtime generation when first
/// polled and fail after shutdown or in a forked child.
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
    /// Returns the owned-I/O capability for the currently polled runtime future.
    ///
    /// A current runtime is installed only while [`crate::Runtime::block_on`] polls
    /// its root future or while a spawned nOS-V task callback polls its future.
    /// Code that already has a runtime or [`crate::runtime::Handle`] should prefer
    /// its explicit `io_handle` method.
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

/// Lazy, resource-safe, single-completion operation future.
///
/// Resources remain local before the first poll. The first poll transfers them to
/// the driver; after that point, progress no longer depends on repolling this
/// future. Drop requests cancellation, while the driver retains resources through
/// the terminal original CQE.
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

/// Converts a Linux CQE result into a byte count or its positive errno error.
pub(crate) fn result(result: i32) -> io::Result<usize> {
    if result < 0 {
        Err(io::Error::from_raw_os_error(-result))
    } else {
        Ok(result as usize)
    }
}

/// Creates the uniform [`io::ErrorKind::BrokenPipe`] runtime-closure error.
pub(crate) fn runtime_closed_error() -> io::Error {
    io::Error::new(io::ErrorKind::BrokenPipe, "nOS-V I/O runtime is closed")
}

/// Creates the uniform [`io::ErrorKind::Unsupported`] opcode-probe error.
pub(crate) fn unsupported_opcode_error() -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        "kernel io_uring does not support this operation",
    )
}

#[cfg(feature = "tokio-compat")]
/// Tokio borrowed-I/O traits and read buffer used by the TCP compatibility layer.
pub use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
