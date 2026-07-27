//! The crate's sole general-purpose nOS-V FFI boundary.

use crate::error::NativeError;
use std::{ffi::CStr, ptr::NonNull};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RawTask(NonNull<nosv_sys::nosv_task>);

// SAFETY: nOS-V documents ordinary create, submit, and destroy as callable from
// everywhere. Higher layers serialize submit against destroy with NativeGate.
unsafe impl Send for RawTask {}
// SAFETY: shared copies are never dereferenced directly and every lifetime-
// sensitive native operation is serialized by NativeGate.
unsafe impl Sync for RawTask {}

impl RawTask {
    pub(crate) unsafe fn from_ptr(pointer: nosv_sys::nosv_task_t) -> Option<Self> {
        NonNull::new(pointer).map(Self)
    }
    pub(crate) const fn as_ptr(self) -> nosv_sys::nosv_task_t {
        self.0.as_ptr()
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RawTaskType(NonNull<nosv_sys::nosv_task_type>);

// SAFETY: task types are immutable after initialization and remain alive until
// shutdown; runtime lifecycle locking prevents use after retirement.
unsafe impl Send for RawTaskType {}
// SAFETY: see the Send justification above.
unsafe impl Sync for RawTaskType {}

impl RawTaskType {
    pub(crate) const fn as_ptr(self) -> nosv_sys::nosv_task_type_t {
        self.0.as_ptr()
    }
}

pub(crate) fn init() -> Result<(), NativeError> {
    // SAFETY: the caller enforces thread-local lifecycle pairing.
    NativeError::from_code(unsafe { nosv_sys::nosv_init() })
}

pub(crate) fn shutdown() -> Result<(), NativeError> {
    // SAFETY: all descriptors have been drained and the owner thread is calling.
    NativeError::from_code(unsafe { nosv_sys::nosv_shutdown() })
}

pub(crate) fn current() -> Option<RawTask> {
    // SAFETY: nosv_self is a read-only TLS query and is valid before attachment.
    unsafe { RawTask::from_ptr(nosv_sys::nosv_self()) }
}

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

pub(crate) fn type_destroy(task_type: RawTaskType) -> Result<(), NativeError> {
    // SAFETY: the runtime has stopped spawning and drained this type's tasks.
    NativeError::from_code(unsafe {
        nosv_sys::nosv_type_destroy(task_type.as_ptr(), nosv_sys::NOSV_TYPE_DESTROY_NONE)
    })
}

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

pub(crate) fn metadata(task: RawTask) -> Result<NonNull<u8>, NativeError> {
    // SAFETY: the task is live under its gate.
    let pointer = unsafe { nosv_sys::nosv_get_task_metadata(task.as_ptr()) };
    NonNull::new(pointer.cast()).ok_or(NativeError::InvalidMetadataSize)
}

pub(crate) fn set_priority(task: RawTask, priority: i32) {
    // SAFETY: only called before first submission.
    unsafe { nosv_sys::nosv_set_task_priority(task.as_ptr(), priority) };
}

pub(crate) fn set_affinity(task: RawTask, affinity: nosv_sys::nosv_affinity_t) {
    let mut affinity = affinity;
    // SAFETY: only called before first submission; the native call copies value.
    unsafe { nosv_sys::nosv_set_task_affinity(task.as_ptr(), &mut affinity) };
}

pub(crate) fn submit(task: RawTask) -> Result<(), NativeError> {
    submit_with(task, nosv_sys::NOSV_SUBMIT_NONE)
}

#[cfg(feature = "time")]
pub(crate) fn submit_deadline_wake(task: RawTask) -> Result<(), NativeError> {
    submit_with(task, nosv_sys::NOSV_SUBMIT_DEADLINE_WAKE)
}

fn submit_with(task: RawTask, flags: nosv_sys::nosv_flags_t) -> Result<(), NativeError> {
    // SAFETY: caller holds the descriptor's gate through this call.
    NativeError::from_code(unsafe { nosv_sys::nosv_submit(task.as_ptr(), flags) })
}

pub(crate) fn suspend() -> Result<(), NativeError> {
    // SAFETY: only invoked by an nOS-V run callback immediately before return.
    NativeError::from_code(unsafe { nosv_sys::nosv_suspend() })
}

pub(crate) fn destroy(task: RawTask) -> Result<(), NativeError> {
    // SAFETY: caller owns the non-joinable descriptor and serializes submission.
    NativeError::from_code(unsafe {
        nosv_sys::nosv_destroy(task.as_ptr(), nosv_sys::NOSV_DESTROY_NONE)
    })
}

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

pub(crate) fn pause() -> Result<(), NativeError> {
    // SAFETY: caller is the attached external task.
    NativeError::from_code(unsafe { nosv_sys::nosv_pause(nosv_sys::NOSV_PAUSE_NONE) })
}

#[cfg(feature = "time")]
pub(crate) fn waitfor(duration: std::time::Duration) -> Result<(), NativeError> {
    let nanos = duration.as_nanos().min(u128::from(u64::MAX)) as u64;
    // SAFETY: caller is a non-parallel nOS-V task. Null is accepted for the
    // optional actual-duration output.
    NativeError::from_code(unsafe { nosv_sys::nosv_waitfor(nanos, std::ptr::null_mut()) })
}

pub(crate) fn detach() -> Result<(), NativeError> {
    // SAFETY: caller is the same thread that attached and no pause is active.
    NativeError::from_code(unsafe { nosv_sys::nosv_detach(nosv_sys::NOSV_DETACH_NONE) })
}

pub(crate) fn current_cpu() -> Result<i32, NativeError> {
    // SAFETY: CurrentTask ensures an nOS-V task context.
    let value = unsafe { nosv_sys::nosv_get_current_system_cpu() };
    if value < 0 {
        Err(NativeError::InvalidOperation)
    } else {
        Ok(value)
    }
}

pub(crate) fn current_numa_node() -> Result<i32, NativeError> {
    // SAFETY: CurrentTask ensures an nOS-V task context.
    let value = unsafe { nosv_sys::nosv_get_current_system_numa_node() };
    if value < 0 {
        Err(NativeError::InvalidOperation)
    } else {
        Ok(value)
    }
}
