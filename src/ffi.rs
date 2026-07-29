//! The crate's sole general-purpose nOS-V FFI boundary.
//!
//! This module converts integer return codes and nullable C pointers into small
//! Rust types. It intentionally does **not** make descriptor operations safe on
//! their own: callers must still satisfy the lifecycle and locking preconditions
//! documented on each function. Keeping those calls here makes the unsafe audit
//! local and prevents `nosv-sys` platform types from leaking into the public API.

use crate::error::NativeError;
use std::{ffi::CStr, ptr::NonNull};

/// A non-null native task descriptor with no ownership semantics of its own.
///
/// The descriptor is copyable because nOS-V scheduling needs the same identity
/// in the runtime registry, callback metadata, and wakers. Its actual lifetime is
/// governed by `task::NativeGate`: every submit and destroy operation is
/// serialized there. Safe public APIs never receive this value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub(crate) struct RawTask(
    /// Non-null pointer returned by nOS-V for a live task descriptor.
    NonNull<nosv_sys::nosv_task>,
);

// SAFETY: nOS-V documents ordinary create, submit, and destroy as callable from
// everywhere. Higher layers serialize submit against destroy with NativeGate.
unsafe impl Send for RawTask {}
// SAFETY: shared copies are never dereferenced directly and every lifetime-
// sensitive native operation is serialized by NativeGate.
unsafe impl Sync for RawTask {}

impl RawTask {
    /// Converts a possibly-null C task handle into the internal non-null wrapper.
    ///
    /// This performs no liveness validation. It is used immediately after a
    /// successful native constructor or at callback entry, where nOS-V's API
    /// contract supplies the required lifetime.
    ///
    /// # Safety
    ///
    /// A non-null `pointer` must designate a nOS-v task that remains valid for
    /// every operation performed through the returned wrapper. The caller must
    /// also serialize descriptor destruction against all other uses.
    pub(crate) unsafe fn from_ptr(pointer: nosv_sys::nosv_task_t) -> Option<Self> {
        NonNull::new(pointer).map(Self)
    }

    /// Returns the original C task pointer for a single audited FFI call.
    ///
    /// This method deliberately does not expose dereferencing. Callers still
    /// need to hold the relevant task or parker gate for lifetime-sensitive
    /// operations.
    pub(crate) const fn as_ptr(self) -> nosv_sys::nosv_task_t {
        self.0.as_ptr()
    }
}

/// A non-null native task-type descriptor owned by a runtime generation.
///
/// Task types are initialized before their tasks and retired only after all of
/// those tasks have completed. nOS-V currently retains type storage until final
/// shutdown, but the wrapper does not rely on callers knowing that detail.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub(crate) struct RawTaskType(
    /// Non-null pointer produced by `nosv_type_init`.
    NonNull<nosv_sys::nosv_task_type>,
);

// SAFETY: task types are immutable after initialization and remain alive until
// shutdown; runtime lifecycle locking prevents use after retirement.
unsafe impl Send for RawTaskType {}
// SAFETY: see the Send justification above.
unsafe impl Sync for RawTaskType {}

impl RawTaskType {
    /// Returns the underlying task-type pointer for an audited FFI operation.
    ///
    /// The caller must ensure that runtime shutdown has not retired this type.
    pub(crate) const fn as_ptr(self) -> nosv_sys::nosv_task_type_t {
        self.0.as_ptr()
    }
}

/// Initializes nOS-V on the calling thread and translates its status code.
///
/// `RuntimeBuilder` calls this before publishing a runtime. nOS-V maintains a
/// thread-local initialization count, so the matching [`shutdown`] call must be
/// made by the same thread after every descriptor belonging to this generation
/// has been drained.
pub(crate) fn init() -> Result<(), NativeError> {
    // SAFETY: the caller enforces thread-local lifecycle pairing.
    NativeError::from_code(unsafe { nosv_sys::nosv_init() })
}

/// Decrements nOS-V's initialization count on the calling owner thread.
///
/// Higher layers close spawning, drain async tasks and drivers, and retire task
/// types before reaching this function. Calling native shutdown earlier could
/// invalidate descriptors still reachable from Rust wakers.
pub(crate) fn shutdown() -> Result<(), NativeError> {
    // SAFETY: all descriptors have been drained and the owner thread is calling.
    NativeError::from_code(unsafe { nosv_sys::nosv_shutdown() })
}

/// Returns the task currently associated with this pthread, if any.
///
/// nOS-V implements this as a thread-local query. The result is used only as a
/// capability check—for example, to reject nested runtime construction—or is
/// immediately scoped by higher-level current-task APIs.
pub(crate) fn current() -> Option<RawTask> {
    // SAFETY: nosv_self is a read-only TLS query and is valid before attachment.
    unsafe { RawTask::from_ptr(nosv_sys::nosv_self()) }
}

/// Creates a native task type for one runtime-owned callback family.
///
/// The wrapper fixes the unsupported end callback and type metadata to null and
/// uses only `NOSV_TYPE_INIT_NONE`. This keeps external-task types, arbitrary
/// flags, and unowned metadata outside the safe executor's state space.
///
/// A successful native status accompanied by a null output is treated as an
/// invariant violation and converted to [`NativeError::InvalidOperation`].
pub(crate) fn type_init(
    run: nosv_sys::nosv_task_run_callback_t,
    completed: nosv_sys::nosv_task_completed_callback_t,
    cost: nosv_sys::nosv_cost_function_t,
    label: &CStr,
) -> Result<RawTaskType, NativeError> {
    let mut raw = std::ptr::null_mut();
    // SAFETY: callbacks have C ABI, label is NUL-terminated and retained only by
    // nOS-V's type initialization contract, and the output is writable.
    NativeError::from_code(unsafe {
        nosv_sys::nosv_type_init(
            &mut raw,
            run,
            None,
            completed,
            label.as_ptr(),
            std::ptr::null_mut(),
            cost,
            nosv_sys::NOSV_TYPE_INIT_NONE,
        )
    })?;
    NonNull::new(raw)
        .map(RawTaskType)
        .ok_or(NativeError::InvalidOperation)
}

/// Retires a runtime-owned native task type after its final task has drained.
///
/// Current nOS-V releases defer the physical reclamation, but invoking the API
/// preserves the intended lifecycle and allows a future implementation to make
/// destruction effective without changing this crate.
pub(crate) fn type_destroy(task_type: RawTaskType) -> Result<(), NativeError> {
    // SAFETY: the runtime has stopped spawning and drained this type's tasks.
    NativeError::from_code(unsafe {
        nosv_sys::nosv_type_destroy(task_type.as_ptr(), nosv_sys::NOSV_TYPE_DESTROY_NONE)
    })
}

/// Allocates an unsubmitted, non-parallel, non-joinable native task.
///
/// The requested metadata region is owned by the descriptor. Executor callers
/// reserve exactly enough bytes for an unaligned `NativeOwner` pointer; the
/// future itself remains in aligned Rust allocation.
pub(crate) fn create(task_type: RawTaskType, metadata_size: usize) -> Result<RawTask, NativeError> {
    let mut raw = std::ptr::null_mut();
    // SAFETY: the task type remains live and the output pointer is writable.
    NativeError::from_code(unsafe {
        nosv_sys::nosv_create(
            &mut raw,
            task_type.as_ptr(),
            metadata_size,
            nosv_sys::NOSV_CREATE_NONE,
        )
    })?;
    // SAFETY: a successful create must return a descriptor.
    unsafe { RawTask::from_ptr(raw) }.ok_or(NativeError::InvalidOperation)
}

/// Obtains the non-null start address of a live task's metadata region.
///
/// The returned pointer carries no alignment guarantee. Callers therefore use
/// `read_unaligned` and `write_unaligned` for the stored owner pointer and never
/// form a reference directly into this region.
pub(crate) fn metadata(task: RawTask) -> Result<NonNull<u8>, NativeError> {
    // SAFETY: the task is live under its gate.
    let pointer = unsafe { nosv_sys::nosv_get_task_metadata(task.as_ptr()) };
    NonNull::new(pointer.cast()).ok_or(NativeError::InvalidMetadataSize)
}

/// Sets native scheduling priority before the task's first submission.
///
/// nOS-V documents concurrent or post-submit priority mutation as undefined, so
/// this function is crate-private and is called only during spawn construction.
pub(crate) fn set_priority(task: RawTask, priority: i32) {
    // SAFETY: only called before first submission.
    unsafe { nosv_sys::nosv_set_task_priority(task.as_ptr(), priority) };
}

/// Copies a validated native affinity into an unsubmitted task descriptor.
///
/// The mutable local exists only because the C signature accepts a mutable
/// pointer; ownership is not transferred and the safe [`crate::Affinity`] value
/// cannot be mutated after spawning.
pub(crate) fn set_affinity(task: RawTask, affinity: nosv_sys::nosv_affinity_t) {
    let mut affinity = affinity;
    // SAFETY: only called before first submission; the native call copies value.
    unsafe { nosv_sys::nosv_set_task_affinity(task.as_ptr(), &mut affinity) };
}

/// Submits or wakes a live task using ordinary nOS-V semantics.
///
/// The caller must hold the descriptor's authoritative gate across this call.
/// That lock makes the native pointer check and the submit operation atomic with
/// respect to completion-time destruction.
pub(crate) fn submit(task: RawTask) -> Result<(), NativeError> {
    submit_with(task, nosv_sys::NOSV_SUBMIT_NONE)
}

/// Interrupts a timer driver's current `nosv_waitfor` deadline.
///
/// Deadline wake is distinct from ordinary submit: nOS-V records an early wake
/// if it arrives just before the driver arms its timeout, preventing the new
/// earlier timer from being lost.
#[cfg(any(feature = "time", feature = "io-uring"))]
pub(crate) fn submit_deadline_wake(task: RawTask) -> Result<(), NativeError> {
    submit_with(task, nosv_sys::NOSV_SUBMIT_DEADLINE_WAKE)
}

/// Performs the common gated native submit operation with an audited flag set.
///
/// Keeping the flag-taking helper private prevents arbitrary combinations such
/// as inline, blocking, or immediate execution from entering the async task
/// state machine without a separate safety proof.
fn submit_with(task: RawTask, flags: nosv_sys::nosv_flags_t) -> Result<(), NativeError> {
    // SAFETY: caller holds the descriptor's gate through this call.
    NativeError::from_code(unsafe { nosv_sys::nosv_submit(task.as_ptr(), flags) })
}

/// Marks the currently running callback to return suspended rather than complete.
///
/// This must be the final meaningful operation on a `Poll::Pending` path. A wake
/// that races before this call is handled by nOS-V's blocking-count handshake;
/// a later callback invocation performs the next Rust poll.
pub(crate) fn suspend() -> Result<(), NativeError> {
    // SAFETY: only invoked by a nOS-v run callback immediately before return.
    NativeError::from_code(unsafe { nosv_sys::nosv_suspend() })
}

/// Destroys a terminal, non-joinable native descriptor.
///
/// Callers hold the same gate used by all submit paths, take the descriptor out
/// of the gate first, and publish Rust join completion only after this returns.
pub(crate) fn destroy(task: RawTask) -> Result<(), NativeError> {
    // SAFETY: caller owns the non-joinable descriptor and serializes submission.
    NativeError::from_code(unsafe {
        nosv_sys::nosv_destroy(task.as_ptr(), nosv_sys::NOSV_DESTROY_NONE)
    })
}

/// Attaches the initialized external owner pthread as a nOS-v task.
///
/// `Runtime::try_block_on` uses the returned descriptor only in a private parker.
/// No affinity or instrumentation flags are requested, and the same pthread must
/// eventually call [`detach`].
pub(crate) fn attach(label: &CStr) -> Result<RawTask, NativeError> {
    let mut raw = std::ptr::null_mut();
    // SAFETY: caller is an initialized external owner thread; output is writable.
    NativeError::from_code(unsafe {
        nosv_sys::nosv_attach(
            &mut raw,
            std::ptr::null_mut(),
            label.as_ptr(),
            nosv_sys::NOSV_ATTACH_NONE,
        )
    })?;
    // SAFETY: successful attach returns the external task descriptor.
    unsafe { RawTask::from_ptr(raw) }.ok_or(NativeError::InvalidOperation)
}

/// Parks the current attached or stackful native task until it is submitted.
///
/// Root `block_on` and the idle timer driver use this instead of blocking a
/// worker pthread on an operating-system condition variable. nOS-V's early-wake
/// counter closes the wake-before-pause race.
pub(crate) fn pause() -> Result<(), NativeError> {
    // SAFETY: caller is the attached external task or internal timer task.
    NativeError::from_code(unsafe { nosv_sys::nosv_pause(nosv_sys::NOSV_PAUSE_NONE) })
}

/// Cooperatively yields the current non-parallel driver task.
#[cfg(feature = "io-uring")]
pub(crate) fn yield_now() -> Result<(), NativeError> {
    // SAFETY: only the dedicated non-parallel I/O driver calls this wrapper.
    NativeError::from_code(unsafe { nosv_sys::nosv_yield(nosv_sys::NOSV_YIELD_NONE) }).map(drop)
}

/// Configures batching for submissions issued by wakers on the current driver task.
#[cfg(feature = "io-uring")]
pub(crate) fn set_submit_window_size(size: usize) -> Result<(), NativeError> {
    // SAFETY: the caller is the running I/O driver and size was validated nonzero.
    NativeError::from_code(unsafe { nosv_sys::nosv_set_submit_window_size(size) })
}

/// Publishes every task submission accumulated in the current submit window.
#[cfg(feature = "io-uring")]
pub(crate) fn flush_submit_window() -> Result<(), NativeError> {
    // SAFETY: the caller is the running I/O driver task.
    NativeError::from_code(unsafe { nosv_sys::nosv_flush_submit_window() })
}

/// Yields the timer driver's native task until a relative duration elapses.
///
/// Nanoseconds saturate at `u64::MAX` to match the C ABI. The driver may be
/// resumed earlier through [`submit_deadline_wake`], which is why this wrapper
/// intentionally uses `nosv_waitfor` instead of timeout-suspend mode.
#[cfg(any(feature = "time", feature = "io-uring"))]
pub(crate) fn waitfor(duration: std::time::Duration) -> Result<(), NativeError> {
    let nanos = duration.as_nanos().min(u128::from(u64::MAX)) as u64;
    // SAFETY: caller is a non-parallel nOS-V task. Null is accepted for the
    // optional actual-duration output.
    NativeError::from_code(unsafe { nosv_sys::nosv_waitfor(nanos, std::ptr::null_mut()) })
}

/// Detaches the external `block_on` pthread from nOS-V.
///
/// The caller closes its parker first, ensuring late cloned root wakers become
/// no-ops before nOS-V destroys the implicit external-task descriptor.
pub(crate) fn detach() -> Result<(), NativeError> {
    // SAFETY: caller is the same thread that attached and no pause is active.
    NativeError::from_code(unsafe { nosv_sys::nosv_detach(nosv_sys::NOSV_DETACH_NONE) })
}

/// Queries the system CPU executing the current nOS-V task.
///
/// Negative native values cannot represent a valid system identifier and are
/// normalized to [`NativeError::InvalidOperation`]. A scoped `CurrentTask`
/// capability ensures this is not called from ordinary external code.
pub(crate) fn current_cpu() -> Result<i32, NativeError> {
    // SAFETY: CurrentTask ensures a nOS-v task context.
    let value = unsafe { nosv_sys::nosv_get_current_system_cpu() };

    if value < 0 {
        Err(NativeError::InvalidOperation)
    } else {
        Ok(value)
    }
}

/// Queries the system NUMA node executing the current nOS-V task.
///
/// Like [`current_cpu`], this is available only through the scoped current-task
/// API and rejects negative sentinel values rather than exposing them as IDs.
pub(crate) fn current_numa_node() -> Result<i32, NativeError> {
    // SAFETY: CurrentTask ensures a nOS-v task context.
    let value = unsafe { nosv_sys::nosv_get_current_system_numa_node() };

    if value < 0 {
        Err(NativeError::InvalidOperation)
    } else {
        Ok(value)
    }
}
