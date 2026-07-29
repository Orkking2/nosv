//! Runtime-wide, futures-based access to raw `io_uring` operations.
//!
//! Each [`crate::Runtime`] owns one ring and one non-parallel nOS-V driver task.
//! Clones of [`IoUringHandle`] share that driver, so callers can submit from
//! concurrent tasks without synchronizing access to the submission queue
//! themselves. The driver serializes SQ access, is the sole CQ consumer, and
//! delivers typed completion entries through [`CompletionStream`].
//!
//! # Submission and completion
//!
//! [`IoUringHandle::submit_entries`] incrementally admits an iterator of raw
//! submission queue entries (SQEs). The admission future resolves only after
//! the iterator is exhausted and every yielded SQE has been accepted by the
//! kernel. CQEs may arrive during admission and are buffered until the resulting
//! stream is consumed.
//!
//! The driver temporarily replaces each SQE's `user_data` with an internal
//! context pointer. Every delivered [`Completion`] restores the caller's value
//! and includes the zero-based index of the SQE in the submitted iterator. CQEs
//! are delivered in arrival order, not iterator order, and a multishot SQE can
//! produce more than one completion with the same index.
//!
//! # Safety and cancellation
//!
//! Raw SQEs erase the lifetimes of buffers, file descriptors, iovecs, socket
//! addresses, and other resources the kernel may access. The caller must keep
//! every resource referenced by an admitted SQE valid until that SQE's terminal
//! CQE has been observed. An `IORING_OP_ASYNC_CANCEL` completion reports only the
//! cancellation request's result; it does **not** prove that the target operation
//! is terminal. After cancellation, use [`Cancellation::wait_drained`] as the
//! point at which referenced resources may be released.
//!
//! Dropping an admission future, completion stream, or cancellation handle
//! requests cancellation as applicable and detaches completion delivery, but it
//! does not synchronously wait for the kernel. Code that relies on drop must
//! keep referenced resources alive by some other means, such as owning them for
//! the remainder of runtime shutdown.
//!
//! # Example
//!
//! ```no_run
//! use nosv::{Runtime, io_uring::raw};
//!
//! let runtime = Runtime::new()?;
//! let ring = runtime.io_uring_handle();
//!
//! runtime.block_on(async move {
//!     let sqe = raw::opcode::Nop::new().build().user_data(7);
//!     // SAFETY: NOP does not reference any external kernel-visible resources.
//!     let mut completions = unsafe { ring.submit_entries([sqe]) }.await.unwrap();
//!     let completion = completions.next().await.unwrap().unwrap();
//!     assert_eq!(completion.index, 0);
//!     assert_eq!(completion.cqe.user_data(), 7);
//! });
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::{ffi, runtime::RuntimeCore, util::lock};
use std::{
    collections::{HashMap, VecDeque},
    error::Error,
    fmt,
    future::Future,
    io,
    marker::PhantomData,
    panic::{self, AssertUnwindSafe},
    pin::Pin,
    ptr,
    sync::{Arc, Condvar, Mutex, PoisonError, Weak},
    task::{Context, Poll, Waker},
    time::Duration,
};
use uring::{IoUring, Probe, cqueue, opcode, squeue};

/// The exact low-level `io-uring` crate used by this runtime.
///
/// Use this re-export to build SQEs whose entry marker matches the
/// [`crate::Runtime`] and [`IoUringHandle`] types. It also prevents callers from
/// accidentally mixing entry types from an incompatible dependency version.
pub use uring as raw;

/// Configuration for the runtime-wide ring and its driver.
///
/// Supply this through [`RuntimeBuilder`](crate::runtime::RuntimeBuilder::io_uring_config).
/// Validation and ring creation happen during runtime construction, before the
/// runtime is published.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IoUringConfig {
    /// Number of entries in the kernel submission queue.
    ///
    /// The value must be non-zero and a power of two.
    pub entries: u32,
    /// Maximum number of CQEs processed in one driver pass.
    ///
    /// Reaching this limit makes the driver yield and schedule another pass
    /// instead of entering its timed wait.
    pub reap_size: usize,
    /// Maximum delay between CQ polls while requests remain in flight.
    ///
    /// Notifications from newly admitted work wake the driver early.
    pub poll_interval: Duration,
    /// Maximum number of undelivered CQEs retained for one submission.
    ///
    /// Original-operation and cancellation CQEs use separate queues with this
    /// limit. An original queue overflow closes normal delivery and requests
    /// cancellation; cancellation overflow does not stop terminal tracking.
    pub max_buffered_completions: usize,
}

impl Default for IoUringConfig {
    /// Returns the general-purpose ring and driver settings.
    ///
    /// The defaults use 256 SQ entries, reap at most 256 CQEs per pass, poll
    /// active work every 50 microseconds, and retain up to 65,536 undelivered
    /// CQEs in each queue belonging to a submission.
    fn default() -> Self {
        Self {
            entries: 256,
            reap_size: 256,
            poll_interval: Duration::from_micros(50),
            max_buffered_completions: 65_536,
        }
    }
}

impl IoUringConfig {
    /// Validates fields required for bounded buffering and driver progress.
    ///
    /// This method has no native side effects and returns the original value on
    /// success.
    ///
    /// # Errors
    ///
    /// Returns the corresponding [`InvalidIoUringConfig`] variant when the ring
    /// depth is zero or not a power of two, or when a reap, polling, or buffering
    /// limit is zero.
    pub fn validate(self) -> Result<Self, InvalidIoUringConfig> {
        if self.entries == 0 {
            return Err(InvalidIoUringConfig::ZeroEntries);
        }
        if !self.entries.is_power_of_two() {
            return Err(InvalidIoUringConfig::EntriesNotPowerOfTwo);
        }
        if self.reap_size == 0 {
            return Err(InvalidIoUringConfig::ZeroReapSize);
        }
        if self.poll_interval.is_zero() {
            return Err(InvalidIoUringConfig::ZeroPollInterval);
        }
        if self.max_buffered_completions == 0 {
            return Err(InvalidIoUringConfig::ZeroCompletionBuffer);
        }
        Ok(self)
    }
}

/// Reason an [`IoUringConfig`] cannot be used to construct a ring.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum InvalidIoUringConfig {
    /// [`IoUringConfig::entries`] is zero.
    ZeroEntries,
    /// [`IoUringConfig::entries`] is not a power of two.
    EntriesNotPowerOfTwo,
    /// [`IoUringConfig::reap_size`] is zero.
    ZeroReapSize,
    /// [`IoUringConfig::poll_interval`] is zero.
    ZeroPollInterval,
    /// [`IoUringConfig::max_buffered_completions`] is zero.
    ZeroCompletionBuffer,
}
impl fmt::Display for InvalidIoUringConfig {
    /// Formats the violated configuration requirement.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ZeroEntries => "io_uring entries must be non-zero",
            Self::EntriesNotPowerOfTwo => "io_uring entries must be a power of two",
            Self::ZeroReapSize => "io_uring reap size must be non-zero",
            Self::ZeroPollInterval => "io_uring poll interval must be non-zero",
            Self::ZeroCompletionBuffer => "io_uring completion buffer limit must be non-zero",
        })
    }
}
impl Error for InvalidIoUringConfig {}

/// One typed CQE and the index of the SQE that produced it.
///
/// Completions arrive in CQ order, so `index` values need not be monotonic. A
/// multishot operation produces multiple values with the same `index`.
#[derive(Clone, Debug)]
pub struct Completion<C> {
    /// Zero-based position of the originating SQE in the submitted iterator.
    pub index: usize,
    /// Typed CQE with the originating SQE's caller-provided `user_data` restored.
    ///
    /// Inspect its result and flags with the APIs in [`raw::cqueue`].
    pub cqe: C,
}

/// Error reported in place of a buffered completion.
///
/// Kernel completions are still consumed and terminal state is still tracked;
/// only delivery of individual CQEs to the caller has overflowed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CompletionError {
    /// A per-submission completion queue filled before it was read.
    ///
    /// Original-completion overflow also closes normal delivery and requests
    /// cancellation. After cancellation-completion overflow,
    /// [`Cancellation::wait_drained`] remains usable.
    BufferOverflow,
}
impl fmt::Display for CompletionError {
    /// Formats the completion-delivery failure.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("io_uring completion buffer overflowed; cancellation was requested")
    }
}
impl Error for CompletionError {}

/// Reason an SQE iterator could not be fully admitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SubmissionErrorKind {
    /// The runtime is closing, closed, unavailable, or inherited through `fork`.
    ///
    /// Some SQEs may already have reached the kernel before admission stopped.
    RuntimeClosed,
    /// Original-CQE buffering overflowed before the iterator was fully admitted.
    CompletionBufferOverflow,
}
impl fmt::Display for SubmissionErrorKind {
    /// Formats the admission failure reason.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::RuntimeClosed => "the io_uring runtime is closed",
            Self::CompletionBufferOverflow => "completion buffering overflowed during admission",
        })
    }
}
impl Error for SubmissionErrorKind {}

/// Failed SQE admission together with the capability to drain accepted work.
///
/// Admission can fail after one or more SQEs have reached the kernel. Recover
/// the [`Cancellation`] with [`SubmissionError::into_cancellation`] and await
/// [`Cancellation::wait_drained`] before releasing referenced resources.
pub struct SubmissionError<S, C>
where
    S: squeue::EntryMarker,
    C: cqueue::EntryMarker,
{
    /// Reason admission stopped.
    kind: SubmissionErrorKind,
    /// Capability that observes cancellation CQEs and terminal retirement.
    cancellation: Cancellation<S, C>,
}
impl<S, C> SubmissionError<S, C>
where
    S: squeue::EntryMarker,
    C: cqueue::EntryMarker,
{
    /// Returns the reason admission stopped without consuming the error.
    pub const fn kind(&self) -> SubmissionErrorKind {
        self.kind
    }
    /// Returns the handle used to inspect cancellation and await retirement.
    ///
    /// Await [`Cancellation::wait_drained`] before releasing resources
    /// referenced by any SQE that might have been admitted.
    pub fn into_cancellation(self) -> Cancellation<S, C> {
        self.cancellation
    }
}
impl<S, C> fmt::Debug for SubmissionError<S, C>
where
    S: squeue::EntryMarker,
    C: cqueue::EntryMarker,
{
    /// Formats the error kind without exposing internal cancellation state.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SubmissionError").field(&self.kind).finish()
    }
}
impl<S, C> fmt::Display for SubmissionError<S, C>
where
    S: squeue::EntryMarker,
    C: cqueue::EntryMarker,
{
    /// Delegates user-facing formatting to the admission failure kind.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.kind.fmt(f)
    }
}
impl<S, C> Error for SubmissionError<S, C>
where
    S: squeue::EntryMarker,
    C: cqueue::EntryMarker,
{
}

/// Controls which completion queues remain visible for one submission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Delivery {
    /// Original CQEs are delivered normally.
    Open,
    /// Explicit cancellation has closed original-CQE delivery.
    Cancelling,
    /// Original-CQE buffering overflowed and cancellation was requested.
    Overflow,
    /// The caller dropped its capability, so no CQEs are retained.
    Detached,
}

/// Shared, independently locked state for one submitted iterator.
///
/// Every live CQE context owns a strong reference, keeping this state allocated
/// until the associated kernel operations retire.
struct SubmissionState<C: cqueue::EntryMarker> {
    /// Mutable delivery queues, counters, and waiter registrations.
    inner: Mutex<SubmissionInner<C>>,
    /// Maximum retained items in each delivery queue.
    limit: usize,
}

/// Mutable counters and delivery state for one submitted iterator.
struct SubmissionInner<C: cqueue::EntryMarker> {
    /// Current caller-visible delivery mode.
    delivery: Delivery,
    /// Original-operation CQEs awaiting stream consumption.
    normal: VecDeque<Completion<C>>,
    /// `AsyncCancel` CQEs awaiting cancellation-stream consumption.
    cancellations: VecDeque<Completion<C>>,
    /// Whether an original completion overflow still needs to be reported.
    normal_overflow: bool,
    /// Whether a cancellation completion overflow still needs to be reported.
    cancel_overflow: bool,
    /// Whether the admission iterator has returned `None`.
    iterator_finished: bool,
    /// Number of original SQEs staged in the userspace SQ.
    originals: usize,
    /// Number of original SQEs accepted by the kernel.
    accepted: usize,
    /// Number of original SQEs whose CQE cleared `IORING_CQE_F_MORE`.
    original_terminal: usize,
    /// Number of cancellation SQEs created for this submission.
    cancels: usize,
    /// Number of cancellation SQEs with a terminal CQE.
    cancel_terminal: usize,
    /// Task waiting for full iterator admission.
    admission_waker: Option<Waker>,
    /// Task waiting for an original CQE or end of stream.
    normal_waker: Option<Waker>,
    /// Task waiting for a cancellation CQE or end of stream.
    cancel_waker: Option<Waker>,
    /// Task waiting for all original and cancellation contexts to retire.
    drain_waker: Option<Waker>,
}
impl<C: cqueue::EntryMarker> SubmissionState<C> {
    /// Allocates zeroed tracking state with the given per-queue delivery limit.
    fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit,
            inner: Mutex::new(SubmissionInner {
                delivery: Delivery::Open,
                normal: VecDeque::new(),
                cancellations: VecDeque::new(),
                normal_overflow: false,
                cancel_overflow: false,
                iterator_finished: false,
                originals: 0,
                accepted: 0,
                original_terminal: 0,
                cancels: 0,
                cancel_terminal: 0,
                admission_waker: None,
                normal_waker: None,
                cancel_waker: None,
                drain_waker: None,
            }),
        })
    }
}

/// Identifies the operation represented by a pointer-backed CQE context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ContextKind {
    /// A caller-provided operation.
    Original,
    /// An `IORING_OP_ASYNC_CANCEL` operation generated by the driver.
    Cancel,
}
/// Tracks whether an SQE context is local, kernel-owned, or terminal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ContextPhase {
    /// The SQE is in the userspace SQ but is not known to be submitted.
    Staged,
    /// The kernel accepted the SQE and may produce CQEs for it.
    Submitted,
    /// The context received its terminal CQE.
    Terminal,
}

/// Stable allocation addressed through the SQE and CQE `user_data` fields.
struct CqeContext<C: cqueue::EntryMarker> {
    /// Whether this context represents original or driver-generated work.
    kind: ContextKind,
    /// Current submission and completion phase.
    phase: ContextPhase,
    /// Iterator index of the associated original SQE.
    index: usize,
    /// Caller `user_data` temporarily replaced by this allocation's address.
    original_user_data: u64,
    /// Submission-level queues, counts, and waiter registrations.
    state: Arc<SubmissionState<C>>,
    /// Original context pointer targeted by a cancellation SQE.
    target: Option<u64>,
    /// Whether cancellation has been requested for an original context.
    cancel_queued: bool,
    /// Whether the cancellation context targeting this original has retired.
    cancel_seen: bool,
}

/// Pending request to create and stage an `AsyncCancel` SQE.
struct CancelCommand<C: cqueue::EntryMarker> {
    /// Stable pointer of the original context to cancel.
    target: u64,
    /// Iterator index copied from the original context.
    index: usize,
    /// Caller `user_data` copied from the original SQE.
    original_user_data: u64,
    /// Submission to which the cancellation result belongs.
    state: Arc<SubmissionState<C>>,
}

/// State serialized across all producers and the sole CQ consumer.
struct DriverData<C: cqueue::EntryMarker> {
    /// Live contexts keyed by the pointer stored in kernel `user_data`.
    contexts: HashMap<u64, Box<CqeContext<C>>>,
    /// Context pointers in SQ order not yet known to be accepted.
    staged: VecDeque<u64>,
    /// Cancellation commands waiting for free SQ slots.
    cancel_queue: VecDeque<CancelCommand<C>>,
    /// Weak registry used to cancel submissions during runtime shutdown.
    submissions: Vec<Weak<SubmissionState<C>>>,
    /// Whether submission futures may admit more original SQEs.
    accepting: bool,
    /// Whether the driver should stop once every work queue is empty.
    shutdown: bool,
}

/// Native scheduling state protected by [`IoUringDriver::gate`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DriverPark {
    /// The native descriptor is being constructed and cannot be woken.
    Building,
    /// The driver callback is executing or scheduled to continue.
    Running,
    /// The idle driver is paused indefinitely.
    Paused,
    /// The active driver is sleeping until its polling deadline.
    DeadlineWait,
    /// A wake submission has been issued but has not resumed the callback.
    WakePending,
    /// The run loop stopped after draining shutdown work.
    Terminal,
    /// The completion callback destroyed the native descriptor.
    Destroyed,
}

/// Native task descriptor and lost-wakeup prevention state.
struct DriverGate {
    /// Driver task descriptor after successful native creation.
    raw: Option<ffi::RawTask>,
    /// Current scheduling phase of the driver task.
    phase: DriverPark,
    /// Work was announced before or while the driver tried to park.
    notified: bool,
}

/// A validated ring prepared before native runtime initialization.
///
/// Building the ring first lets [`RuntimeBuilder`](crate::runtime::RuntimeBuilder) reject local
/// configuration or kernel capability errors without partly initializing nOS-V.
pub(crate) struct PreparedRing<S: squeue::EntryMarker, C: cqueue::EntryMarker> {
    /// Kernel ring created with the selected entry marker types.
    ring: IoUring<S, C>,
    /// Validated scheduling and buffering policy retained by the driver.
    config: IoUringConfig,
}

impl<S: squeue::EntryMarker, C: cqueue::EntryMarker> PreparedRing<S, C> {
    /// Validates configuration, creates the ring, and probes required features.
    ///
    /// The ring is marked `MADV_DONTFORK` by the low-level builder. Construction
    /// also requires `IORING_FEAT_NODROP` and `IORING_OP_ASYNC_CANCEL`, because
    /// lossless CQ consumption and explicit retirement are driver invariants.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidIoUringConfig`](crate::InitError::InvalidIoUringConfig) for invalid settings,
    /// or [`IoUring`](crate::InitError::IoUring) when creation, probing, or a required
    /// kernel capability fails.
    pub(crate) fn new(config: IoUringConfig) -> Result<Self, crate::InitError> {
        let config = config
            .validate()
            .map_err(crate::InitError::InvalidIoUringConfig)?;

        let mut builder = IoUring::<S, C>::builder();
        builder.dontfork();

        let ring = builder
            .build(config.entries)
            .map_err(crate::InitError::IoUring)?;

        if !ring.params().is_feature_nodrop() {
            return Err(crate::InitError::IoUring(unsupported(
                "kernel io_uring does not provide IORING_FEAT_NODROP",
            )));
        }

        let mut probe = Probe::new();
        ring.submitter()
            .register_probe(&mut probe)
            .map_err(crate::InitError::IoUring)?;

        if !probe.is_supported(opcode::AsyncCancel::CODE) {
            return Err(crate::InitError::IoUring(unsupported(
                "kernel io_uring does not support IORING_OP_ASYNC_CANCEL",
            )));
        }

        Ok(Self { ring, config })
    }
}

/// Thread-safe submission capability for one runtime's typed ring.
///
/// A handle borrows no runtime thread state and may be cloned or moved between
/// tasks. All clones submit to the same ring. The type parameters preserve the
/// SQE and CQE widths selected by the owning [`crate::Runtime`].
///
/// A handle becomes closed when its runtime begins shutdown, is dropped, or is
/// observed from a child created with `fork`.
pub struct IoUringHandle<S = squeue::Entry, C = cqueue::Entry>
where
    S: squeue::EntryMarker,
    C: cqueue::EntryMarker,
{
    /// Shared ring owner and CQ consumer.
    driver: Arc<IoUringDriver<S, C>>,
    /// Weak runtime core used to reject closed or fork-inherited access.
    core: Weak<RuntimeCore>,
}
impl<S: squeue::EntryMarker, C: cqueue::EntryMarker> Clone for IoUringHandle<S, C> {
    /// Clones the capability without creating another kernel ring.
    fn clone(&self) -> Self {
        Self {
            driver: self.driver.clone(),
            core: self.core.clone(),
        }
    }
}
impl<S, C> IoUringHandle<S, C>
where
    S: squeue::EntryMarker + Send + 'static,
    C: cqueue::EntryMarker + Send + 'static,
{
    /// Creates a handle tied to `core` and sharing `driver`.
    pub(crate) fn new(driver: Arc<IoUringDriver<S, C>>, core: &Arc<RuntimeCore>) -> Self {
        Self {
            driver,
            core: Arc::downgrade(core),
        }
    }

    /// Creates a future that incrementally admits an iterator of raw SQEs.
    ///
    /// Polling the returned [`SubmitEntries`] fills available SQ slots and wakes
    /// the shared driver. It resolves to a [`CompletionStream`] only after the
    /// iterator ends and all yielded SQEs have been accepted. The iterator is not
    /// eagerly collected, so it can be much larger than [`IoUringConfig::entries`].
    ///
    /// If admission stops after accepting a prefix, [`SubmissionError`] retains
    /// the cancellation and drain capability for that prefix. Dropping the
    /// admission future also requests cancellation, but detaches delivery and
    /// provides no synchronous drain guarantee.
    ///
    /// # Safety
    ///
    /// For each SQE accepted by the kernel, all referenced memory, descriptors,
    /// registrations, and other kernel-visible resources must remain valid until
    /// its terminal original CQE. This remains true if admission fails, the
    /// future or stream is dropped, or cancellation is requested.
    ///
    /// On successful admission, consuming the stream through `None` without
    /// observing a [`CompletionError`] proves all original operations terminal.
    /// After such an error, or to end early, call
    /// [`CompletionStream::cancel`] and await [`Cancellation::wait_drained`]
    /// before invalidating referenced resources. On failed admission, obtain the
    /// cancellation handle with [`SubmissionError::into_cancellation`] and apply
    /// the same rule.
    ///
    /// The caller must also satisfy every opcode-specific safety contract from
    /// the low-level [`raw`] crate. In particular, this API does not take
    /// ownership of resources merely because the SQE value itself is owned.
    pub unsafe fn submit_entries<I>(&self, entries: I) -> SubmitEntries<I::IntoIter, S, C>
    where
        I: IntoIterator<Item = S>,
    {
        let state = self.driver.register_submission();

        let closed = self
            .core
            .upgrade()
            .is_none_or(|core| core.ensure_running().is_err());
        if closed {
            lock(&state.inner).delivery = Delivery::Cancelling;
        }
        SubmitEntries {
            iterator: Some(entries.into_iter()),
            pending: None,
            next_index: 0,
            driver: self.driver.clone(),
            state,
            closed,
            done: false,
        }
    }
}

/// Future that admits an SQE iterator into the shared submission queue.
///
/// Completion CQEs can arrive while admission is still in progress and count
/// against [`IoUringConfig::max_buffered_completions`]. If the future is dropped,
/// the iterator and any not-yet-staged SQE are dropped, while already staged or
/// submitted operations are cancelled and their completion delivery is detached.
/// The unsafe resource-lifetime contract from [`IoUringHandle::submit_entries`]
/// continues to apply to those operations.
#[must_use = "submission starts only when this future is polled"]
pub struct SubmitEntries<I, S, C>
where
    I: Iterator<Item = S>,
    S: squeue::EntryMarker + Send + 'static,
    C: cqueue::EntryMarker + Send + 'static,
{
    /// Source iterator until admission finishes or fails.
    iterator: Option<I>,
    /// SQE removed from the iterator but not yet staged because the SQ was full.
    pending: Option<(usize, S)>,
    /// Iterator index assigned to the next yielded SQE.
    next_index: usize,
    /// Shared driver used to stage entries and request cancellation.
    driver: Arc<IoUringDriver<S, C>>,
    /// Completion queues, counters, and waiter registrations for this iterator.
    state: Arc<SubmissionState<C>>,
    /// Whether the originating runtime was already unavailable at construction.
    closed: bool,
    /// Whether the future has returned its one terminal result.
    done: bool,
}
impl<I, S, C> Future for SubmitEntries<I, S, C>
where
    I: Iterator<Item = S>,
    S: squeue::EntryMarker + Send + 'static,
    C: cqueue::EntryMarker + Send + 'static,
{
    /// A stream for fully admitted work, or a failure retaining drain capability.
    type Output = Result<CompletionStream<S, C>, SubmissionError<S, C>>;

    /// Advances iterator admission and registers the admission waiter.
    ///
    /// # Panics
    ///
    /// Panics if polled again after returning `Ready`, as permitted by the
    /// [`Future`] contract. A panic produced by the user's iterator also
    /// propagates; subsequent drop still requests cancellation for admitted SQEs.
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: no field is structurally pinned and the iterator is never moved out while pinned.
        let this = unsafe { self.get_unchecked_mut() };
        assert!(!this.done, "SubmitEntries polled after completion");
        if !this.closed {
            let iterator = this.iterator.as_mut().expect("live admission iterator");
            this.driver.admit(
                iterator,
                &mut this.pending,
                &mut this.next_index,
                &this.state,
            );
        }
        let mut state = lock(&this.state.inner);
        let failure = if state.delivery == Delivery::Overflow {
            Some(SubmissionErrorKind::CompletionBufferOverflow)
        } else if this.closed
            || state.delivery == Delivery::Cancelling
            || state.delivery == Delivery::Detached
        {
            Some(SubmissionErrorKind::RuntimeClosed)
        } else {
            None
        };
        if let Some(kind) = failure {
            this.done = true;
            this.iterator.take();
            this.pending.take();
            drop(state);
            this.driver
                .request_cancel(&this.state, Delivery::Cancelling);
            return Poll::Ready(Err(SubmissionError {
                kind,
                cancellation: Cancellation::new(this.driver.clone(), this.state.clone()),
            }));
        }
        if state.iterator_finished && state.accepted == state.originals {
            this.done = true;
            this.iterator.take();
            this.pending.take();
            drop(state);
            return Poll::Ready(Ok(CompletionStream {
                driver: this.driver.clone(),
                state: this.state.clone(),
                active: true,
                _sqe: PhantomData,
            }));
        }
        replace_waker(&mut state.admission_waker, cx.waker());
        drop(state);
        this.driver.notify();
        Poll::Pending
    }
}
impl<I, S, C> Drop for SubmitEntries<I, S, C>
where
    I: Iterator<Item = S>,
    S: squeue::EntryMarker + Send + 'static,
    C: cqueue::EntryMarker + Send + 'static,
{
    /// Cancels and detaches already staged work when admission is abandoned.
    fn drop(&mut self) {
        if !self.done {
            self.driver.request_cancel(&self.state, Delivery::Detached);
        }
    }
}

/// Arrival-ordered original CQEs for one fully admitted iterator.
///
/// The stream owns the right to receive original-operation completions. If no
/// [`CompletionError`] occurs, draining it through `None` proves normal terminal
/// completion. After an error, consume it with [`CompletionStream::cancel`] and
/// await the returned retirement capability.
/// Dropping it instead requests cancellation and detaches all delivery without
/// waiting for referenced resources to become safe to release.
pub struct CompletionStream<S, C>
where
    S: squeue::EntryMarker + Send + 'static,
    C: cqueue::EntryMarker + Send + 'static,
{
    /// Shared driver used for explicit or drop-triggered cancellation.
    driver: Arc<IoUringDriver<S, C>>,
    /// Original-completion queue and terminal counters for this submission.
    state: Arc<SubmissionState<C>>,
    /// Whether drop still needs to cancel and detach this stream.
    active: bool,
    /// Associates the stream with its SQE marker without implying ownership.
    _sqe: PhantomData<fn() -> S>,
}
impl<S, C> CompletionStream<S, C>
where
    S: squeue::EntryMarker + Send + 'static,
    C: cqueue::EntryMarker + Send + 'static,
{
    /// Returns a future for the next original-operation completion.
    ///
    /// The future returns `Some(Ok(_))` for a retained CQE, `Some(Err(_))` when
    /// delivery overflowed, and `None` after normal terminal completion or once
    /// normal delivery has closed. Each call must finish or be dropped before
    /// calling `next` again because it borrows the stream mutably.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> NextCompletion<'_, C> {
        NextCompletion { state: &self.state }
    }
    /// Closes original-CQE delivery and requests cancellation of live work.
    ///
    /// One `IORING_OP_ASYNC_CANCEL` request is queued for each non-terminal
    /// original context, including operations staged but not yet known to be
    /// accepted. Any original CQEs already buffered by this stream cease to be
    /// accessible. Use the returned [`Cancellation`] to inspect cancellation
    /// results and to await actual operation retirement.
    pub fn cancel(mut self) -> Cancellation<S, C> {
        self.active = false;
        self.driver
            .request_cancel(&self.state, Delivery::Cancelling);
        Cancellation::new(self.driver.clone(), self.state.clone())
    }
}
impl<S, C> Drop for CompletionStream<S, C>
where
    S: squeue::EntryMarker + Send + 'static,
    C: cqueue::EntryMarker + Send + 'static,
{
    /// Requests cancellation and detaches delivery when the stream is abandoned.
    fn drop(&mut self) {
        if self.active {
            self.driver.request_cancel(&self.state, Delivery::Detached);
        }
    }
}

/// Future returned by [`CompletionStream::next`].
///
/// This future borrows its stream and stores at most one task waker in the
/// submission. Dropping it leaves buffered completions available to a later
/// call to [`CompletionStream::next`].
pub struct NextCompletion<'a, C: cqueue::EntryMarker> {
    /// Submission state containing the original-CQE queue.
    state: &'a Arc<SubmissionState<C>>,
}
impl<C: cqueue::EntryMarker> Future for NextCompletion<'_, C> {
    /// One completion or overflow error, or `None` when delivery has ended.
    type Output = Option<Result<Completion<C>, CompletionError>>;

    /// Pops a buffered CQE, reports pending overflow, or registers the waiter.
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = lock(&self.state.inner);
        if let Some(cqe) = state.normal.pop_front() {
            return Poll::Ready(Some(Ok(cqe)));
        }
        if state.normal_overflow {
            state.normal_overflow = false;
            return Poll::Ready(Some(Err(CompletionError::BufferOverflow)));
        }
        if state.delivery != Delivery::Open
            || (state.iterator_finished && state.original_terminal == state.originals)
        {
            return Poll::Ready(None);
        }
        replace_waker(&mut state.normal_waker, cx.waker());
        Poll::Pending
    }
}

/// Cancellation results and terminal-retirement tracking for one submission.
///
/// Creating this value means normal original-CQE delivery has closed and the
/// driver has queued `IORING_OP_ASYNC_CANCEL` for each live original context.
/// Cancellation is inherently racy: an operation can complete normally before
/// its cancellation request reaches it. Inspect raw cancellation CQE results
/// with [`Cancellation::next`], but use [`Cancellation::wait_drained`]—not any
/// individual result—as the resource-lifetime boundary.
///
/// Dropping this handle detaches delivery and does not wait for retirement.
pub struct Cancellation<S, C>
where
    S: squeue::EntryMarker,
    C: cqueue::EntryMarker,
{
    /// Strong reference keeping the driver available while cancellation drains.
    driver: Arc<IoUringDriver<S, C>>,
    /// Cancellation-CQE queue and terminal counts for this submission.
    state: Arc<SubmissionState<C>>,
    /// Whether drop still needs to detach caller-visible queues.
    active: bool,
}
impl<S: squeue::EntryMarker, C: cqueue::EntryMarker> Cancellation<S, C> {
    /// Creates an active cancellation capability for `state`.
    fn new(driver: Arc<IoUringDriver<S, C>>, state: Arc<SubmissionState<C>>) -> Self {
        Self {
            driver,
            state,
            active: true,
        }
    }
    /// Returns a future for the next `AsyncCancel` CQE.
    ///
    /// Each delivered [`Completion`] uses the index and restored `user_data` of
    /// the targeted original SQE. The raw CQE result describes the cancellation
    /// request and may indicate that the target already completed or could not be
    /// found. `None` means every cancellation request itself is terminal; it does
    /// not replace [`Cancellation::wait_drained`] for resource retirement.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> NextCancellation<'_, C> {
        NextCancellation { state: &self.state }
    }
    /// Returns a future that waits for every associated context to be terminal.
    ///
    /// Once this future returns, all original SQEs have produced a terminal CQE
    /// and all driver-generated cancellation SQEs have completed. Resources
    /// covered by the unsafe submission contract can then be released. Awaiting
    /// this does not discard cancellation CQEs already buffered for [`Self::next`].
    pub fn wait_drained(&mut self) -> WaitDrained<'_, C> {
        WaitDrained { state: &self.state }
    }
}
impl<S: squeue::EntryMarker, C: cqueue::EntryMarker> Drop for Cancellation<S, C> {
    /// Detaches caller-visible completion queues without blocking for the kernel.
    fn drop(&mut self) {
        if self.active {
            let _keep_driver_alive = &self.driver;
            detach(&self.state);
        }
    }
}

/// Future returned by [`Cancellation::next`].
///
/// Dropping this future leaves retained cancellation CQEs available to a later
/// call to [`Cancellation::next`].
pub struct NextCancellation<'a, C: cqueue::EntryMarker> {
    /// Submission state containing the cancellation-CQE queue.
    state: &'a Arc<SubmissionState<C>>,
}
impl<C: cqueue::EntryMarker> Future for NextCancellation<'_, C> {
    /// One cancellation CQE or overflow error, or `None` when all are terminal.
    type Output = Option<Result<Completion<C>, CompletionError>>;

    /// Pops a buffered CQE, reports overflow, or registers the cancellation waiter.
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = lock(&self.state.inner);
        if let Some(cqe) = state.cancellations.pop_front() {
            return Poll::Ready(Some(Ok(cqe)));
        }
        if state.cancel_overflow {
            state.cancel_overflow = false;
            return Poll::Ready(Some(Err(CompletionError::BufferOverflow)));
        }
        if state.cancel_terminal == state.cancels {
            return Poll::Ready(None);
        }
        replace_waker(&mut state.cancel_waker, cx.waker());
        Poll::Pending
    }
}

/// Future returned by [`Cancellation::wait_drained`].
///
/// This future is independent of completion buffering: it becomes ready from
/// terminal counters even if delivery overflowed or the individual CQEs were not
/// consumed.
pub struct WaitDrained<'a, C: cqueue::EntryMarker> {
    /// Submission state containing original and cancellation terminal counts.
    state: &'a Arc<SubmissionState<C>>,
}
impl<C: cqueue::EntryMarker> Future for WaitDrained<'_, C> {
    /// No value; readiness itself certifies terminal retirement.
    type Output = ();

    /// Checks terminal counts or registers the drain waiter.
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let mut state = lock(&self.state.inner);
        if drained(&state) {
            Poll::Ready(())
        } else {
            replace_waker(&mut state.drain_waker, cx.waker());
            Poll::Pending
        }
    }
}

/// Runtime-wide ring owner, serialized SQ producer, and sole CQ consumer.
///
/// Submission futures may call into this object concurrently. `data` serializes
/// their SQ access with the native driver callback, while only that callback
/// reads the CQ. The native task is non-parallel, so at most one run loop is
/// active. Shutdown stops admission, cancels all registered submissions, drains
/// every context, then lets the completion callback destroy native ownership.
pub(crate) struct IoUringDriver<S: squeue::EntryMarker, C: cqueue::EntryMarker> {
    /// Context registry, pending queues, and admission/shutdown flags.
    data: Mutex<DriverData<C>>,
    /// Ring removed exactly once when the driver reaches terminal shutdown.
    ring: Mutex<Option<IoUring<S, C>>>,
    /// Validated queue, polling, and buffering policy.
    config: IoUringConfig,
    /// Native scheduling phase and lost-wakeup state.
    gate: Mutex<DriverGate>,
    /// Whether the native completion callback has retired the driver task.
    completed: Mutex<bool>,
    /// Notifies the owner thread waiting in [`Self::shutdown_and_wait`].
    completed_cv: Condvar,
}

impl<S, C> IoUringDriver<S, C>
where
    S: squeue::EntryMarker + Send + 'static,
    C: cqueue::EntryMarker + Send + 'static,
{
    /// Creates and initially submits the native task that owns `prepared`.
    ///
    /// A raw strong `Arc` pointer is stored in pointer-sized task metadata. The
    /// native completion callback is its sole consumer after successful initial
    /// submission.
    ///
    /// # Errors
    ///
    /// Returns a translated native error if task creation, metadata access, or
    /// initial submission fails. Every partially created native or Rust resource
    /// is released before the error is returned.
    pub(crate) fn start(
        task_type: ffi::RawTaskType,
        prepared: PreparedRing<S, C>,
    ) -> Result<Arc<Self>, crate::NativeError> {
        let driver = Arc::new(Self {
            data: Mutex::new(DriverData {
                contexts: HashMap::new(),
                staged: VecDeque::new(),
                cancel_queue: VecDeque::new(),
                submissions: Vec::new(),
                accepting: true,
                shutdown: false,
            }),
            ring: Mutex::new(Some(prepared.ring)),
            config: prepared.config,
            gate: Mutex::new(DriverGate {
                raw: None,
                phase: DriverPark::Building,
                notified: false,
            }),
            completed: Mutex::new(false),
            completed_cv: Condvar::new(),
        });

        let raw = ffi::create(task_type, std::mem::size_of::<*const Self>())?;

        let metadata = match ffi::metadata(raw) {
            Ok(value) => value,
            Err(error) => {
                let _ = ffi::destroy(raw);
                return Err(error);
            }
        };

        let owner = Arc::into_raw(driver.clone());
        // SAFETY: metadata is pointer-sized and completion is the sole consumer.
        unsafe { ptr::write_unaligned(metadata.as_ptr().cast::<*const Self>(), owner) };

        {
            let mut gate = lock(&driver.gate);
            gate.raw = Some(raw);
            gate.phase = DriverPark::Running;
        }

        if let Err(error) = ffi::submit(raw) {
            {
                let mut gate = lock(&driver.gate);
                gate.raw = None;
                gate.phase = DriverPark::Destroyed;
            }

            let _ = ffi::destroy(raw);
            // SAFETY: no callback can consume the failed initial submission.
            unsafe { drop(Arc::from_raw(owner)) };

            return Err(error);
        }

        Ok(driver)
    }

    /// Allocates submission state and adds it to the shutdown registry.
    ///
    /// Dead weak entries are removed opportunistically so completed submissions
    /// do not make registry growth unbounded.
    fn register_submission(&self) -> Arc<SubmissionState<C>> {
        let state = SubmissionState::new(self.config.max_buffered_completions);

        let mut data = lock(&self.data);
        data.submissions.retain(|weak| weak.strong_count() != 0);
        data.submissions.push(Arc::downgrade(&state));
        state
    }

    /// Fills available SQ slots from an iterator and submits the staged prefix.
    ///
    /// Before admitting original work, this method gives queued cancellations
    /// priority. Each original receives a stable context pointer in place of its
    /// caller `user_data`; the original value is retained for CQE restoration.
    /// An SQE that does not fit remains in `pending` with its assigned index for
    /// the next poll.
    fn admit<I: Iterator<Item = S>>(
        &self,
        iterator: &mut I,
        pending: &mut Option<(usize, S)>,
        next_index: &mut usize,
        state: &Arc<SubmissionState<C>>,
    ) {
        let mut wakes = Vec::new();
        let mut notify = false;
        {
            let mut data = lock(&self.data);
            if !data.accepting {
                drop(data);
                self.request_cancel(state, Delivery::Cancelling);
                return;
            }
            let mut ring_guard = lock(&self.ring);
            let ring = ring_guard
                .as_mut()
                .unwrap_or_else(|| invariant_abort("ring missing during admission"));
            // SAFETY: the driver data mutex serializes every SQ producer.
            let mut sq = unsafe { ring.submission_shared() };
            sq.sync();
            stage_cancellations::<S, C>(&mut data, &mut sq);
            loop {
                if pending.is_none() {
                    match iterator.next() {
                        Some(entry) => {
                            let index = *next_index;
                            *next_index = next_index
                                .checked_add(1)
                                .unwrap_or_else(|| invariant_abort("SQE index exhausted"));
                            *pending = Some((index, entry));
                        }
                        None => {
                            let mut submission = lock(&state.inner);
                            submission.iterator_finished = true;
                            if let Some(waker) = submission.admission_waker.take() {
                                wakes.push(waker);
                            }
                            break;
                        }
                    }
                }
                let (index, mut entry) = pending.take().expect("pending SQE");
                let original_user_data = entry.get_user_data();
                let boxed = Box::new(CqeContext {
                    kind: ContextKind::Original,
                    phase: ContextPhase::Staged,
                    index,
                    original_user_data,
                    state: state.clone(),
                    target: None,
                    cancel_queued: false,
                    cancel_seen: false,
                });
                let pointer = context_pointer(&boxed);
                if data.contexts.contains_key(&pointer) {
                    invariant_abort("allocator reused a live context pointer");
                }
                entry.set_user_data(pointer);
                // SAFETY: the caller's submit_entries contract retains all SQE resources.
                if unsafe { sq.push(&entry) }.is_err() {
                    entry.set_user_data(original_user_data);
                    *pending = Some((index, entry));
                    break;
                }
                data.contexts.insert(pointer, boxed);
                data.staged.push_back(pointer);
                lock(&state.inner).originals += 1;
                notify = true;
            }
            drop(sq);
            submit_staged(&mut data, ring, &mut wakes);
        }
        wake_waiters(wakes);
        if notify || pending.is_some() {
            self.notify();
        }
    }

    /// Closes delivery as requested and queues cancellation for live originals.
    ///
    /// At most one cancellation command is created for each original context.
    /// Detached delivery additionally discards both retained CQE queues and
    /// their consumer wakers, but terminal counters and the drain waker remain.
    fn request_cancel(&self, state: &Arc<SubmissionState<C>>, delivery: Delivery) {
        {
            let mut submission = lock(&state.inner);
            if submission.delivery == Delivery::Open || delivery == Delivery::Detached {
                submission.delivery = delivery;
            }
            if delivery == Delivery::Detached {
                submission.normal.clear();
                submission.cancellations.clear();
                submission.normal_waker = None;
                submission.cancel_waker = None;
            }
        }
        let mut data = lock(&self.data);
        let targets = data
            .contexts
            .iter()
            .filter_map(|(&pointer, context)| {
                (context.kind == ContextKind::Original
                    && context.phase != ContextPhase::Terminal
                    && !context.cancel_queued
                    && Arc::ptr_eq(&context.state, state))
                .then_some(pointer)
            })
            .collect::<Vec<_>>();
        let mut commands = Vec::with_capacity(targets.len());
        for pointer in targets {
            let context = data.contexts.get_mut(&pointer).expect("collected context");
            context.cancel_queued = true;
            commands.push(CancelCommand {
                target: pointer,
                index: context.index,
                original_user_data: context.original_user_data,
                state: context.state.clone(),
            });
        }
        let count = commands.len();
        data.cancel_queue.extend(commands);
        if count != 0 {
            lock(&state.inner).cancels += count;
        }
        drop(data);
        self.notify();
    }

    /// Announces work and wakes a driver that is paused or in a deadline wait.
    ///
    /// `notified` closes the race between a producer finding the driver running
    /// and the driver deciding to park. `WakePending` coalesces further wakeups
    /// until the already submitted native wake resumes the callback.
    fn notify(&self) {
        let mut gate = lock(&self.gate);
        gate.notified = true;
        let deadline = match gate.phase {
            DriverPark::Paused => Some(false),
            DriverPark::DeadlineWait => Some(true),
            DriverPark::Building | DriverPark::Terminal | DriverPark::Destroyed => return,
            DriverPark::Running | DriverPark::WakePending => None,
        };
        if let Some(deadline) = deadline {
            gate.phase = DriverPark::WakePending;
            let raw = gate
                .raw
                .unwrap_or_else(|| invariant_abort("driver descriptor missing"));
            let result = if deadline {
                ffi::submit_deadline_wake(raw)
            } else {
                ffi::submit(raw)
            };
            if result.is_err() {
                invariant_abort("driver wake submit failed");
            }
        }
    }

    /// Runs driver passes until work is drained, or parks according to demand.
    ///
    /// Wakers are invoked through an nOS-V submit window so tasks awakened by a
    /// CQ batch can be made runnable together. A shutdown decision drops the ring
    /// before marking the native task terminal.
    fn run(&self) {
        loop {
            let (wakes, backlog) = self.pass();

            if !wakes.is_empty() {
                ffi::set_submit_window_size(wakes.len())
                    .unwrap_or_else(|_| invariant_abort("submit window set failed"));
                wake_waiters(wakes);
                ffi::flush_submit_window()
                    .unwrap_or_else(|_| invariant_abort("submit window flush failed"));
            }
            let decision = decide(&lock(&self.data), backlog);
            if decision == RunDecision::Stop {
                drop(lock(&self.ring).take());
                lock(&self.gate).phase = DriverPark::Terminal;
                return;
            }
            let should_park = {
                let mut gate = lock(&self.gate);
                if gate.notified {
                    gate.notified = false;
                    gate.phase = DriverPark::Running;
                    false
                } else {
                    gate.phase = match decision {
                        RunDecision::Yield => DriverPark::Running,
                        RunDecision::Wait => DriverPark::DeadlineWait,
                        RunDecision::Pause => DriverPark::Paused,
                        RunDecision::Stop => unreachable!(),
                    };
                    true
                }
            };
            if !should_park {
                continue;
            }
            let result = match decision {
                RunDecision::Yield => ffi::yield_now(),
                RunDecision::Wait => ffi::waitfor(self.config.poll_interval),
                RunDecision::Pause => ffi::pause(),
                RunDecision::Stop => unreachable!(),
            };
            if result.is_err() {
                invariant_abort("driver could not yield or park");
            }
            lock(&self.gate).phase = DriverPark::Running;
        }
    }

    /// Performs one cancellation, submission, and bounded-completion pass.
    ///
    /// Returns the Rust task wakers collected by dispatch and whether the CQ may
    /// still contain a backlog. Original-buffer overflow is converted into
    /// cancellation only after CQ iteration releases the ring and driver locks.
    fn pass(&self) -> (Vec<Waker>, bool) {
        let mut wakes = Vec::new();

        {
            let mut data = lock(&self.data);
            let mut ring_guard = lock(&self.ring);
            let ring = ring_guard
                .as_mut()
                .unwrap_or_else(|| invariant_abort("ring missing during submit"));
            // SAFETY: the data mutex serializes every SQ producer.
            let mut sq = unsafe { ring.submission_shared() };
            sq.sync();
            stage_cancellations::<S, C>(&mut data, &mut sq);
            drop(sq);
            submit_staged(&mut data, ring, &mut wakes);
        }

        let (cqes, backlog) = {
            let ring_guard = lock(&self.ring);
            let ring = ring_guard
                .as_ref()
                .unwrap_or_else(|| invariant_abort("ring missing during reap"));
            // SAFETY: only the non-parallel driver consumes this CQ.
            let mut cq = unsafe { ring.completion_shared() };
            cq.sync();
            let values = cq.by_ref().take(self.config.reap_size).collect::<Vec<_>>();
            let backlog = values.len() == self.config.reap_size || !cq.is_empty();
            (values, backlog)
        };

        let mut overflow = Vec::new();
        for cqe in cqes {
            if let Some(state) = self.dispatch(cqe, &mut wakes) {
                overflow.push(state);
            }
        }
        for state in overflow {
            self.request_cancel(&state, Delivery::Overflow);
        }
        (wakes, backlog)
    }

    /// Routes one typed CQE through its pointer context and updates retirement.
    ///
    /// Caller `user_data` is restored before delivery. Original multishot
    /// contexts remain registered while `IORING_CQE_F_MORE` is set. An original
    /// context targeted by cancellation remains allocated until both its own
    /// terminal CQE and the cancellation CQE have arrived, in either order.
    ///
    /// Returns the submission state only when original delivery first overflows;
    /// the caller then requests cancellation outside this dispatch operation.
    fn dispatch(&self, mut cqe: C, wakes: &mut Vec<Waker>) -> Option<Arc<SubmissionState<C>>> {
        let pointer = cqe.user_data();

        if pointer == 0 {
            invariant_abort("null CQE context pointer");
        }

        let common: cqueue::Entry = cqe.clone().into();
        let more = cqueue::more(common.flags());
        let mut data = lock(&self.data);
        let mut context = data
            .contexts
            .remove(&pointer)
            .unwrap_or_else(|| invariant_abort("unknown or duplicate CQE pointer"));

        if context.phase != ContextPhase::Submitted {
            invariant_abort("CQE for unsubmitted context");
        }

        cqe.set_user_data(context.original_user_data);
        match context.kind {
            ContextKind::Original => {
                let mut overflow = None;
                let mut state = lock(&context.state.inner);
                if !state.iterator_finished {
                    if let Some(waker) = state.admission_waker.take() {
                        wakes.push(waker);
                    }
                }
                if state.delivery == Delivery::Open {
                    if state.normal.len() < context.state.limit {
                        state.normal.push_back(Completion {
                            index: context.index,
                            cqe,
                        });
                    } else if !state.normal_overflow {
                        state.normal_overflow = true;
                        state.delivery = Delivery::Overflow;
                        overflow = Some(context.state.clone());
                    }
                    if let Some(waker) = state.normal_waker.take() {
                        wakes.push(waker);
                    }
                }
                if !more {
                    context.phase = ContextPhase::Terminal;
                    state.original_terminal += 1;
                    if state.original_terminal == state.originals {
                        if let Some(waker) = state.normal_waker.take() {
                            wakes.push(waker);
                        }
                    }
                    wake_drain(&mut state, wakes);
                }
                drop(state);
                if more || (context.cancel_queued && !context.cancel_seen) {
                    data.contexts.insert(pointer, context);
                }
                overflow
            }
            ContextKind::Cancel => {
                if more {
                    invariant_abort("multishot cancellation CQE");
                }
                let target = context
                    .target
                    .unwrap_or_else(|| invariant_abort("cancel context without target"));
                let mut state = lock(&context.state.inner);
                if state.delivery != Delivery::Detached {
                    if state.cancellations.len() < context.state.limit {
                        state.cancellations.push_back(Completion {
                            index: context.index,
                            cqe,
                        });
                    } else {
                        state.cancel_overflow = true;
                    }
                    if let Some(waker) = state.cancel_waker.take() {
                        wakes.push(waker);
                    }
                }
                state.cancel_terminal += 1;
                if state.cancel_terminal == state.cancels {
                    if let Some(waker) = state.cancel_waker.take() {
                        wakes.push(waker);
                    }
                }
                wake_drain(&mut state, wakes);
                drop(state);
                let original = data
                    .contexts
                    .get_mut(&target)
                    .unwrap_or_else(|| invariant_abort("cancel target disappeared"));
                original.cancel_seen = true;
                if original.phase == ContextPhase::Terminal {
                    data.contexts.remove(&target);
                }
                None
            }
        }
    }

    /// Stops admission, cancels registered work, and blocks until task retirement.
    ///
    /// This owner-thread operation does not return until every original and
    /// cancellation context has reached a terminal CQE, the ring has been
    /// dropped, and the native completion callback has destroyed the task.
    pub(crate) fn shutdown_and_wait(&self) {
        let submissions = {
            let mut data = lock(&self.data);
            data.accepting = false;
            data.shutdown = true;
            data.submissions
                .iter()
                .filter_map(Weak::upgrade)
                .collect::<Vec<_>>()
        };
        for state in submissions {
            self.request_cancel(&state, Delivery::Detached);
        }
        self.notify();
        let mut completed = lock(&self.completed);
        while !*completed {
            completed = self
                .completed_cv
                .wait(completed)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Destroys the terminal native descriptor and releases shutdown waiters.
    ///
    /// This is called exactly once by [`completed_callback`], after [`Self::run`]
    /// has drained shutdown work and moved the gate to [`DriverPark::Terminal`].
    fn retire(&self) {
        {
            let mut gate = lock(&self.gate);
            if gate.phase != DriverPark::Terminal {
                invariant_abort("early driver completion");
            }
            let raw = gate
                .raw
                .take()
                .unwrap_or_else(|| invariant_abort("missing driver descriptor"));
            ffi::destroy(raw).unwrap_or_else(|_| invariant_abort("driver destroy failed"));
            gate.phase = DriverPark::Destroyed;
        }
        *lock(&self.completed) = true;
        self.completed_cv.notify_all();
    }
}

/// Converts queued cancellation commands into SQEs until the SQ becomes full.
///
/// Each generated SQE receives its own pointer context while targeting the
/// stable pointer of an original context. A command that does not fit is restored
/// to the front of the queue so cancellation ordering is preserved.
fn stage_cancellations<S: squeue::EntryMarker, C: cqueue::EntryMarker>(
    data: &mut DriverData<C>,
    sq: &mut uring::SubmissionQueue<'_, S>,
) {
    while let Some(command) = data.cancel_queue.pop_front() {
        let boxed = Box::new(CqeContext {
            kind: ContextKind::Cancel,
            phase: ContextPhase::Staged,
            index: command.index,
            original_user_data: command.original_user_data,
            state: command.state.clone(),
            target: Some(command.target),
            cancel_queued: false,
            cancel_seen: false,
        });
        let pointer = context_pointer(&boxed);
        if data.contexts.contains_key(&pointer) {
            invariant_abort("reused live cancel pointer");
        }
        let mut entry = S::from(opcode::AsyncCancel::new(command.target).build());
        entry.set_user_data(pointer);
        // SAFETY: the target context remains allocated until both CQEs are terminal.
        if unsafe { sq.push(&entry) }.is_err() {
            data.cancel_queue.push_front(command);
            break;
        }
        data.contexts.insert(pointer, boxed);
        data.staged.push_back(pointer);
    }
}

/// Submits the current SQ and advances the kernel-accepted context prefix.
///
/// Successful submission counts correspond to the ordered `staged` queue.
/// Original admission counters and wakers are updated only for that accepted
/// prefix. Retryable transient failures leave all contexts staged; any other
/// error violates the runtime's ability to make safe progress and aborts.
fn submit_staged<S: squeue::EntryMarker, C: cqueue::EntryMarker>(
    data: &mut DriverData<C>,
    ring: &IoUring<S, C>,
    wakes: &mut Vec<Waker>,
) {
    match ring.submit() {
        Ok(count) => {
            if count > data.staged.len() {
                invariant_abort("kernel accepted unstaged SQEs");
            }
            for _ in 0..count {
                let pointer = data.staged.pop_front().expect("validated staged prefix");
                let context = data
                    .contexts
                    .get_mut(&pointer)
                    .unwrap_or_else(|| invariant_abort("staged context disappeared"));
                if context.phase != ContextPhase::Staged {
                    invariant_abort("context submitted twice");
                }
                context.phase = ContextPhase::Submitted;
                if context.kind == ContextKind::Original {
                    let mut state = lock(&context.state.inner);
                    state.accepted += 1;
                    if let Some(waker) = state.admission_waker.take() {
                        wakes.push(waker);
                    }
                }
            }
        }
        Err(error) if is_retryable(&error) => {}
        Err(error) => {
            eprintln!("io_uring submission failed: {error}");
            invariant_abort("unrecoverable submit error");
        }
    }
}

/// Scheduling action selected after a driver pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RunDecision {
    /// Shutdown is requested and no kernel or queued work remains.
    Stop,
    /// Work is immediately available, so yield without an idle delay.
    Yield,
    /// Operations are in flight, so wait until the next polling deadline.
    Wait,
    /// No work exists, so pause until a producer notification.
    Pause,
}

/// Chooses how the driver should proceed from queue state and CQ backlog.
fn decide<C: cqueue::EntryMarker>(data: &DriverData<C>, backlog: bool) -> RunDecision {
    let active =
        !data.contexts.is_empty() || !data.staged.is_empty() || !data.cancel_queue.is_empty();

    if data.shutdown && !active {
        RunDecision::Stop
    } else if backlog || !data.staged.is_empty() || !data.cancel_queue.is_empty() {
        RunDecision::Yield
    } else if active {
        RunDecision::Wait
    } else {
        RunDecision::Pause
    }
}

/// Encodes a stable CQE context address for the kernel `user_data` field.
fn context_pointer<C: cqueue::EntryMarker>(context: &CqeContext<C>) -> u64 {
    context as *const CqeContext<C> as usize as u64
}

/// Reports whether every original and generated cancellation is terminal.
fn drained<C: cqueue::EntryMarker>(state: &SubmissionInner<C>) -> bool {
    state.original_terminal == state.originals && state.cancel_terminal == state.cancels
}

/// Moves the registered drain waker to `wakes` once both terminal counts match.
fn wake_drain<C: cqueue::EntryMarker>(state: &mut SubmissionInner<C>, wakes: &mut Vec<Waker>) {
    if drained(state) {
        if let Some(waker) = state.drain_waker.take() {
            wakes.push(waker);
        }
    }
}

/// Marks a submission detached and discards caller-visible CQE delivery.
///
/// Terminal counters and the drain waker are deliberately preserved so internal
/// retirement and an already-created [`WaitDrained`] future remain coherent.
fn detach<C: cqueue::EntryMarker>(state: &SubmissionState<C>) {
    let mut state = lock(&state.inner);
    state.delivery = Delivery::Detached;
    state.normal.clear();
    state.cancellations.clear();
    state.normal_waker = None;
    state.cancel_waker = None;
}

/// Stores a cloned waker unless the existing one wakes the same task.
fn replace_waker(slot: &mut Option<Waker>, replacement: &Waker) {
    if slot.as_ref().is_none_or(|old| !old.will_wake(replacement)) {
        *slot = Some(replacement.clone());
    }
}

/// Classifies transient ring-submission errors that a later pass may resolve.
fn is_retryable(error: &io::Error) -> bool {
    matches!(
        error.raw_os_error(),
        Some(libc::EINTR | libc::EAGAIN | libc::EBUSY)
    )
}

/// Wakes a collected batch while containing panics from foreign waker code.
///
/// A panicking waker must not unwind into the native callback or prevent other
/// completions in the same batch from being delivered.
fn wake_waiters(wakes: Vec<Waker>) {
    for wake in wakes {
        let _ = panic::catch_unwind(AssertUnwindSafe(|| wake.wake()));
    }
}

/// Constructs an [`io::ErrorKind::Unsupported`] capability-probe error.
fn unsupported(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, message)
}

/// Reads the driver owner pointer stored in native task metadata.
///
/// # Safety
///
/// `raw` must be the live task created by [`IoUringDriver::start`] for exactly
/// the entry marker types `S` and `C`. Its metadata must still contain the raw
/// strong pointer installed by that method.
unsafe fn driver_owner<S: squeue::EntryMarker, C: cqueue::EntryMarker>(
    raw: ffi::RawTask,
) -> *const IoUringDriver<S, C> {
    let metadata =
        ffi::metadata(raw).unwrap_or_else(|_| invariant_abort("driver metadata missing"));
    // SAFETY: start wrote this stable pointer and completion owns its strong count.
    unsafe { ptr::read_unaligned(metadata.as_ptr().cast::<*const IoUringDriver<S, C>>()) }
}

/// Native body callback that drives submission and completion passes.
///
/// Panics are contained and converted into process aborts because unwinding over
/// the C ABI would be undefined and abandoning pointer contexts would be unsafe.
///
/// # Safety
///
/// nOS-V must call this with the live driver task created for entry marker types
/// `S` and `C`, and must honor the task type's non-parallel execution contract.
pub(crate) unsafe extern "C" fn run_callback<S, C>(pointer: nosv_sys::nosv_task_t)
where
    S: squeue::EntryMarker + Send + 'static,
    C: cqueue::EntryMarker + Send + 'static,
{
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: nOS-V supplies the live descriptor.
        let raw = unsafe { ffi::RawTask::from_ptr(pointer) }
            .unwrap_or_else(|| invariant_abort("null driver"));
        // SAFETY: metadata retains the owner until completion.
        let driver = unsafe { driver_owner::<S, C>(raw).as_ref() }
            .unwrap_or_else(|| invariant_abort("null owner"));
        driver.run();
    }));
    if result.is_err() {
        invariant_abort("panic in io_uring driver callback");
    }
}

/// Native completion callback that consumes task metadata ownership.
///
/// This callback reconstructs the raw driver `Arc`, retires the native
/// descriptor, and wakes the owner thread blocked in shutdown. Panics abort for
/// the same ABI and ownership reasons as [`run_callback`].
///
/// # Safety
///
/// nOS-V must invoke this exactly once for the terminal driver task created for
/// `S` and `C`, after its body callback has returned from the terminal phase.
pub(crate) unsafe extern "C" fn completed_callback<S, C>(pointer: nosv_sys::nosv_task_t)
where
    S: squeue::EntryMarker + Send + 'static,
    C: cqueue::EntryMarker + Send + 'static,
{
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: nOS-V supplies the terminal descriptor.
        let raw = unsafe { ffi::RawTask::from_ptr(pointer) }
            .unwrap_or_else(|| invariant_abort("null completed driver"));
        // SAFETY: this callback is the sole strong-count consumer.
        let owner = unsafe { Arc::from_raw(driver_owner::<S, C>(raw)) };
        owner.retire();
    }));
    if result.is_err() {
        invariant_abort("panic in io_uring completion callback");
    }
}

/// Reports an internal safety invariant violation and aborts the process.
///
/// These paths cannot unwind or return an ordinary error because doing so could
/// abandon kernel-visible pointers or violate native ownership accounting.
fn invariant_abort(message: &str) -> ! {
    eprintln!("nOS-V io_uring invariant failed: {message}");
    std::process::abort()
}

#[cfg(test)]
/// Deterministic tests for configuration, pointer contexts, and CQ dispatch.
mod tests {
    use super::*;
    use static_assertions::assert_impl_all;

    /// Verifies every invalid configuration field and the valid defaults.
    #[test]
    fn configuration_validation() {
        assert!(IoUringConfig::default().validate().is_ok());
        assert_eq!(
            IoUringConfig {
                entries: 0,
                ..IoUringConfig::default()
            }
            .validate(),
            Err(InvalidIoUringConfig::ZeroEntries)
        );
        assert_eq!(
            IoUringConfig {
                entries: 3,
                ..IoUringConfig::default()
            }
            .validate(),
            Err(InvalidIoUringConfig::EntriesNotPowerOfTwo)
        );
        assert_eq!(
            IoUringConfig {
                reap_size: 0,
                ..IoUringConfig::default()
            }
            .validate(),
            Err(InvalidIoUringConfig::ZeroReapSize)
        );
        assert_eq!(
            IoUringConfig {
                poll_interval: Duration::ZERO,
                ..IoUringConfig::default()
            }
            .validate(),
            Err(InvalidIoUringConfig::ZeroPollInterval)
        );
        assert_eq!(
            IoUringConfig {
                max_buffered_completions: 0,
                ..IoUringConfig::default()
            }
            .validate(),
            Err(InvalidIoUringConfig::ZeroCompletionBuffer)
        );
    }

    /// Verifies that a boxed context has a usable pointer and retains user data.
    #[test]
    fn pointer_context_preserves_metadata() {
        let state = SubmissionState::<cqueue::Entry>::new(1);
        let context = Box::new(CqeContext {
            kind: ContextKind::Original,
            phase: ContextPhase::Staged,
            index: 9,
            original_user_data: u64::MAX,
            state,
            target: None,
            cancel_queued: false,
            cancel_seen: false,
        });
        assert_ne!(context_pointer(&context), 0);
        assert_eq!(context.original_user_data, u64::MAX);
    }

    /// Verifies idle, backlog, and fully drained shutdown scheduling decisions.
    #[test]
    fn park_decisions() {
        let mut data = DriverData::<cqueue::Entry> {
            contexts: HashMap::new(),
            staged: VecDeque::new(),
            cancel_queue: VecDeque::new(),
            submissions: Vec::new(),
            accepting: true,
            shutdown: false,
        };
        assert_eq!(decide(&data, false), RunDecision::Pause);
        assert_eq!(decide(&data, true), RunDecision::Yield);
        data.shutdown = true;
        assert_eq!(decide(&data, false), RunDecision::Stop);
    }
    /// Minimal C-layout CQE prefix used to construct deterministic test entries.
    #[repr(C)]
    struct FakeCqe {
        /// Context pointer or caller value carried in the CQE.
        user_data: u64,
        /// Kernel-style completion result.
        result: i32,
        /// Kernel CQE flag bits.
        flags: u32,
    }

    /// Constructs a typed CQE with a controlled pointer, result, and flags.
    fn fake_cqe(user_data: u64, result: i32, flags: u32) -> cqueue::Entry {
        assert_eq!(
            std::mem::size_of::<FakeCqe>(),
            std::mem::size_of::<cqueue::Entry>()
        );
        // SAFETY: FakeCqe mirrors the kernel io_uring_cqe prefix wrapped by Entry.
        unsafe {
            std::mem::transmute(FakeCqe {
                user_data,
                result,
                flags,
            })
        }
    }

    /// Builds a ringless driver for direct state-machine dispatch tests.
    fn model_driver(
        data: DriverData<cqueue::Entry>,
    ) -> IoUringDriver<squeue::Entry, cqueue::Entry> {
        IoUringDriver {
            data: Mutex::new(data),
            ring: Mutex::new(None),
            config: IoUringConfig::default(),
            gate: Mutex::new(DriverGate {
                raw: None,
                phase: DriverPark::Building,
                notified: false,
            }),
            completed: Mutex::new(false),
            completed_cv: Condvar::new(),
        }
    }

    /// Verifies original CQE routing, metadata restoration, and retirement.
    #[test]
    fn original_dispatch_restores_user_data_and_retires_context() {
        let state = SubmissionState::new(4);
        {
            let mut inner = lock(&state.inner);
            inner.iterator_finished = true;
            inner.originals = 1;
            inner.accepted = 1;
        }
        let context = Box::new(CqeContext {
            kind: ContextKind::Original,
            phase: ContextPhase::Submitted,
            index: 3,
            original_user_data: 0x55aa,
            state: state.clone(),
            target: None,
            cancel_queued: false,
            cancel_seen: false,
        });
        let pointer = context_pointer(&context);
        let mut contexts = HashMap::new();
        contexts.insert(pointer, context);
        let driver = model_driver(DriverData {
            contexts,
            staged: VecDeque::new(),
            cancel_queue: VecDeque::new(),
            submissions: Vec::new(),
            accepting: true,
            shutdown: false,
        });
        let mut wakes = Vec::new();
        assert!(
            driver
                .dispatch(fake_cqe(pointer, -libc::EIO, 0), &mut wakes)
                .is_none()
        );
        assert!(lock(&driver.data).contexts.is_empty());
        let mut inner = lock(&state.inner);
        let completion = inner.normal.pop_front().unwrap();
        assert_eq!(completion.index, 3);
        assert_eq!(completion.cqe.user_data(), 0x55aa);
        assert_eq!(completion.cqe.result(), -libc::EIO);
        assert_eq!(inner.original_terminal, 1);
    }

    /// Exercises dispatch with cancellation and original CQEs in either order.
    fn cancellation_order(cancel_first: bool) {
        let state = SubmissionState::new(4);
        {
            let mut inner = lock(&state.inner);
            inner.delivery = Delivery::Cancelling;
            inner.iterator_finished = true;
            inner.originals = 1;
            inner.accepted = 1;
            inner.cancels = 1;
        }
        let original = Box::new(CqeContext {
            kind: ContextKind::Original,
            phase: ContextPhase::Submitted,
            index: 1,
            original_user_data: 99,
            state: state.clone(),
            target: None,
            cancel_queued: true,
            cancel_seen: false,
        });
        let original_pointer = context_pointer(&original);
        let cancel = Box::new(CqeContext {
            kind: ContextKind::Cancel,
            phase: ContextPhase::Submitted,
            index: 1,
            original_user_data: 99,
            state: state.clone(),
            target: Some(original_pointer),
            cancel_queued: false,
            cancel_seen: false,
        });
        let cancel_pointer = context_pointer(&cancel);
        let mut contexts = HashMap::new();
        contexts.insert(original_pointer, original);
        contexts.insert(cancel_pointer, cancel);
        let driver = model_driver(DriverData {
            contexts,
            staged: VecDeque::new(),
            cancel_queue: VecDeque::new(),
            submissions: Vec::new(),
            accepting: true,
            shutdown: false,
        });
        let mut wakes = Vec::new();
        if cancel_first {
            driver.dispatch(fake_cqe(cancel_pointer, 0, 0), &mut wakes);
            driver.dispatch(fake_cqe(original_pointer, -libc::ECANCELED, 0), &mut wakes);
        } else {
            driver.dispatch(fake_cqe(original_pointer, -libc::ECANCELED, 0), &mut wakes);
            assert!(lock(&driver.data).contexts.contains_key(&original_pointer));
            driver.dispatch(fake_cqe(cancel_pointer, 0, 0), &mut wakes);
        }
        assert!(lock(&driver.data).contexts.is_empty());
        let mut inner = lock(&state.inner);
        assert_eq!(inner.original_terminal, 1);
        assert_eq!(inner.cancel_terminal, 1);
        let completion = inner.cancellations.pop_front().unwrap();
        assert_eq!(completion.index, 1);
        assert_eq!(completion.cqe.user_data(), 99);
    }

    /// Verifies that both legal cancellation-CQE arrival orders retire contexts.
    #[test]
    fn cancellation_cqes_are_order_independent() {
        cancellation_order(false);
        cancellation_order(true);
    }

    /// Models concurrent detach and terminal completion with one destruction.
    #[test]
    fn loom_completion_and_detach_destroy_only_at_terminal_cqe() {
        loom::model(|| {
            use loom::sync::{
                Arc as LoomArc, Mutex as LoomMutex,
                atomic::{AtomicUsize, Ordering},
            };
            use loom::thread;
            let state = LoomArc::new(LoomMutex::new((false, false)));
            let destroys = LoomArc::new(AtomicUsize::new(0));
            let dropped = state.clone();
            let drop_thread = thread::spawn(move || {
                dropped.lock().unwrap().0 = true;
            });
            let completed = state.clone();
            let completed_destroys = destroys.clone();
            let complete_thread = thread::spawn(move || {
                let mut state = completed.lock().unwrap();
                assert!(!state.1);
                state.1 = true;
                completed_destroys.fetch_add(1, Ordering::SeqCst);
            });
            drop_thread.join().unwrap();
            complete_thread.join().unwrap();
            assert!(state.lock().unwrap().1);
            assert_eq!(destroys.load(Ordering::SeqCst), 1);
        });
    }

    /// Verifies multishot contexts survive until `IORING_CQE_F_MORE` clears.
    #[test]
    fn multishot_context_is_retained_until_more_flag_clears() {
        const CQE_F_MORE: u32 = 1 << 1;
        assert!(cqueue::more(CQE_F_MORE));
        let state = SubmissionState::new(4);
        {
            let mut inner = lock(&state.inner);
            inner.iterator_finished = true;
            inner.originals = 1;
            inner.accepted = 1;
        }
        let context = Box::new(CqeContext {
            kind: ContextKind::Original,
            phase: ContextPhase::Submitted,
            index: 0,
            original_user_data: 77,
            state: state.clone(),
            target: None,
            cancel_queued: false,
            cancel_seen: false,
        });
        let pointer = context_pointer(&context);
        let mut contexts = HashMap::new();
        contexts.insert(pointer, context);
        let driver = model_driver(DriverData {
            contexts,
            staged: VecDeque::new(),
            cancel_queue: VecDeque::new(),
            submissions: Vec::new(),
            accepting: true,
            shutdown: false,
        });
        let mut wakes = Vec::new();
        driver.dispatch(fake_cqe(pointer, 1, CQE_F_MORE), &mut wakes);
        assert!(lock(&driver.data).contexts.contains_key(&pointer));
        assert_eq!(lock(&state.inner).original_terminal, 0);
        driver.dispatch(fake_cqe(pointer, 2, 0), &mut wakes);
        assert!(lock(&driver.data).contexts.is_empty());
        let mut inner = lock(&state.inner);
        assert_eq!(inner.original_terminal, 1);
        assert_eq!(inner.normal.len(), 2);
        assert_eq!(inner.normal.pop_front().unwrap().cqe.user_data(), 77);
        assert_eq!(inner.normal.pop_front().unwrap().cqe.user_data(), 77);
    }

    /// Verifies bounded original delivery closes and reports the overflow state.
    #[test]
    fn completion_limit_closes_delivery_and_reports_overflow_once() {
        let state = SubmissionState::new(1);
        {
            let mut inner = lock(&state.inner);
            inner.iterator_finished = true;
            inner.originals = 2;
            inner.accepted = 2;
        }
        let mut contexts = HashMap::new();
        let mut pointers = Vec::new();
        for index in 0..2 {
            let context = Box::new(CqeContext {
                kind: ContextKind::Original,
                phase: ContextPhase::Submitted,
                index,
                original_user_data: index as u64,
                state: state.clone(),
                target: None,
                cancel_queued: false,
                cancel_seen: false,
            });
            let pointer = context_pointer(&context);
            pointers.push(pointer);
            contexts.insert(pointer, context);
        }
        let driver = model_driver(DriverData {
            contexts,
            staged: VecDeque::new(),
            cancel_queue: VecDeque::new(),
            submissions: Vec::new(),
            accepting: true,
            shutdown: false,
        });
        let mut wakes = Vec::new();
        assert!(
            driver
                .dispatch(fake_cqe(pointers[0], 0, 0), &mut wakes)
                .is_none()
        );
        let overflow = driver
            .dispatch(fake_cqe(pointers[1], 0, 0), &mut wakes)
            .expect("overflow state");
        assert!(Arc::ptr_eq(&overflow, &state));
        let inner = lock(&state.inner);
        assert_eq!(inner.delivery, Delivery::Overflow);
        assert!(inner.normal_overflow);
        assert_eq!(inner.normal.len(), 1);
        assert_eq!(inner.original_terminal, 2);
    }

    assert_impl_all!(IoUringHandle<squeue::Entry, cqueue::Entry>: Send, Sync, Clone);
}
