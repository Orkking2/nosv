//! Errors reported by the safe runtime.

use std::{any::Any, error::Error, fmt};

/// A stable Rust representation of a nOS-v status code.
///
/// Known negative values receive descriptive variants; unrecognized values are
/// preserved by [`NativeError::Unknown`] so newer native libraries do not lose
/// diagnostic information. The wrapper avoids relying on nOS-V's incomplete
/// error-string table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NativeError {
    /// A callback was invalid.
    InvalidCallback,
    /// Task metadata was invalid.
    InvalidMetadataSize,
    /// The operation is invalid in the current state.
    InvalidOperation,
    /// An argument was invalid.
    InvalidParameter,
    /// nOS-V is not initialized on this thread.
    NotInitialized,
    /// Native allocation failed.
    OutOfMemory,
    /// a nOS-v task context is required.
    OutsideTask,
    /// A native resource is busy.
    Busy,
    /// A timed native operation expired.
    Timeout,
    /// A code unknown to this crate version.
    Unknown(i32),
}

impl NativeError {
    /// Converts a C return code into Rust's success-or-error convention.
    ///
    /// Zero becomes `Ok(())`; every other value is mapped to a known variant
    /// or preserved verbatim. Centralizing this translation keeps raw integer
    /// comparisons out of lifecycle and executor code.
    pub(crate) fn from_code(code: i32) -> Result<(), Self> {
        if code == nosv_sys::NOSV_SUCCESS {
            return Ok(());
        }
        Err(match code {
            nosv_sys::NOSV_ERR_INVALID_CALLBACK => Self::InvalidCallback,
            nosv_sys::NOSV_ERR_INVALID_METADATA_SIZE => Self::InvalidMetadataSize,
            nosv_sys::NOSV_ERR_INVALID_OPERATION => Self::InvalidOperation,
            nosv_sys::NOSV_ERR_INVALID_PARAMETER => Self::InvalidParameter,
            nosv_sys::NOSV_ERR_NOT_INITIALIZED => Self::NotInitialized,
            nosv_sys::NOSV_ERR_OUT_OF_MEMORY => Self::OutOfMemory,
            nosv_sys::NOSV_ERR_OUTSIDE_TASK => Self::OutsideTask,
            nosv_sys::NOSV_ERR_BUSY => Self::Busy,
            nosv_sys::NOSV_ERR_TIMEOUT => Self::Timeout,
            other => Self::Unknown(other),
        })
    }

    /// Returns the exact integer status represented by this value.
    ///
    /// For [`NativeError::Unknown`] this round-trips the original code, which is
    /// useful for logging and compatibility with native diagnostics.
    pub fn code(self) -> i32 {
        match self {
            Self::InvalidCallback => nosv_sys::NOSV_ERR_INVALID_CALLBACK,
            Self::InvalidMetadataSize => nosv_sys::NOSV_ERR_INVALID_METADATA_SIZE,
            Self::InvalidOperation => nosv_sys::NOSV_ERR_INVALID_OPERATION,
            Self::InvalidParameter => nosv_sys::NOSV_ERR_INVALID_PARAMETER,
            Self::NotInitialized => nosv_sys::NOSV_ERR_NOT_INITIALIZED,
            Self::OutOfMemory => nosv_sys::NOSV_ERR_OUT_OF_MEMORY,
            Self::OutsideTask => nosv_sys::NOSV_ERR_OUTSIDE_TASK,
            Self::Busy => nosv_sys::NOSV_ERR_BUSY,
            Self::Timeout => nosv_sys::NOSV_ERR_TIMEOUT,
            Self::Unknown(code) => code,
        }
    }
}

impl fmt::Display for NativeError {
    /// Formats the error without calling back into nOS-V.
    ///
    /// Keeping formatting purely in Rust makes errors safe to display after the
    /// runtime has shut down or in a forked child.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidCallback => "invalid callback",
            Self::InvalidMetadataSize => "invalid metadata size",
            Self::InvalidOperation => "invalid native operation",
            Self::InvalidParameter => "invalid parameter",
            Self::NotInitialized => "nOS-V is not initialized",
            Self::OutOfMemory => "native allocation failed",
            Self::OutsideTask => "operation requires a nOS-v task",
            Self::Busy => "native resource is busy",
            Self::Timeout => "native operation timed out",
            Self::Unknown(_) => "unknown native error",
        };
        write!(f, "{message} (code {})", self.code())
    }
}

impl Error for NativeError {}

/// Failure while constructing and publishing a runtime generation.
///
/// Initialization is rolled back before this error is returned, so no partially
/// usable [`crate::Runtime`] escapes.
#[derive(Debug)]
#[non_exhaustive]
pub enum InitError {
    /// Runtime construction was attempted from inside a nOS-V task.
    AlreadyInTask,
    /// nOS-V rejected initialization.
    Native(NativeError),
}

impl fmt::Display for InitError {
    /// Formats the error without calling back into nOS-V.
    ///
    /// Keeping formatting purely in Rust makes errors safe to display after the
    /// runtime has shut down or in a forked child.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyInTask => f.write_str("cannot create a runtime inside a nOS-v task"),
            Self::Native(error) => write!(f, "runtime initialization failed: {error}"),
        }
    }
}

impl Error for InitError {
    /// Returns the underlying native error when this variant wraps one.
    ///
    /// State-validation variants originate in the safe layer and therefore have
    /// no lower-level source.
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Native(error) => Some(error),
            _ => None,
        }
    }
}

impl From<NativeError> for InitError {
    /// Wraps a translated native failure for `?`-based propagation.
    fn from(value: NativeError) -> Self {
        Self::Native(value)
    }
}

/// The runtime is no longer accepting operations.
///
/// This zero-sized error intentionally reveals no raw lifecycle state. It is used
/// by capability-style queries where "closed, fork-inherited, or unavailable" is
/// the only safe distinction callers need.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeClosed;

impl fmt::Display for RuntimeClosed {
    /// Formats the error without calling back into nOS-V.
    ///
    /// Keeping formatting purely in Rust makes errors safe to display after the
    /// runtime has shut down or in a forked child.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the nOS-V runtime is closed")
    }
}

impl Error for RuntimeClosed {}

/// Failure to create and initially submit a spawned future.
///
/// On return, the future and any native descriptor allocated for it have been
/// reclaimed; a failed spawn never leaves a detached Rust task behind.
#[derive(Debug)]
#[non_exhaustive]
pub enum SpawnError {
    /// The runtime is closing or closed.
    RuntimeClosed,
    /// The handle was inherited across `fork`.
    ForkedProcess,
    /// Native construction or submission failed.
    Native(NativeError),
}

impl From<RuntimeClosed> for SpawnError {
    fn from(_: RuntimeClosed) -> Self {
        Self::RuntimeClosed
    }
}

impl fmt::Display for SpawnError {
    /// Formats the error without calling back into nOS-V.
    ///
    /// Keeping formatting purely in Rust makes errors safe to display after the
    /// runtime has shut down or in a forked child.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RuntimeClosed => RuntimeClosed::fmt(&RuntimeClosed, f),
            Self::ForkedProcess => f.write_str("runtime handle was inherited across fork"),
            Self::Native(error) => write!(f, "could not spawn task: {error}"),
        }
    }
}

impl Error for SpawnError {
    /// Returns the underlying native error when this variant wraps one.
    ///
    /// State-validation variants originate in the safe layer and therefore have
    /// no lower-level source.
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Native(error) => Some(error),
            _ => None,
        }
    }
}

impl From<NativeError> for SpawnError {
    /// Wraps a translated native failure for `?`-based propagation.
    fn from(value: NativeError) -> Self {
        Self::Native(value)
    }
}

/// Failure while establishing or operating an attached-thread `block_on`.
///
/// User-future panics are resumed after the thread has been detached and are not
/// represented by this type. These variants describe runtime setup and native
/// lifecycle failures only.
#[derive(Debug)]
#[non_exhaustive]
pub enum BlockOnError {
    /// Only the creating thread may attach.
    WrongThread,
    /// `block_on` calls may not be nested.
    Nested,
    /// The caller is already a nOS-v task.
    AlreadyInTask,
    /// The runtime was inherited across `fork`.
    ForkedProcess,
    /// The runtime is closing or closed.
    RuntimeClosed,
    /// A native operation failed.
    Native(NativeError),
}

impl From<RuntimeClosed> for BlockOnError {
    fn from(_: RuntimeClosed) -> Self {
        Self::RuntimeClosed
    }
}

impl fmt::Display for BlockOnError {
    /// Formats the error without calling back into nOS-V.
    ///
    /// Keeping formatting purely in Rust makes errors safe to display after the
    /// runtime has shut down or in a forked child.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongThread => f.write_str("block_on must run on the runtime owner thread"),
            Self::Nested => f.write_str("nested block_on is not supported"),
            Self::AlreadyInTask => f.write_str("block_on cannot run inside a nOS-v task"),
            Self::ForkedProcess => f.write_str("runtime was inherited across fork"),
            Self::RuntimeClosed => RuntimeClosed::fmt(&RuntimeClosed, f),
            Self::Native(error) => write!(f, "block_on native operation failed: {error}"),
        }
    }
}

impl Error for BlockOnError {}

/// Failure while draining and shutting a runtime down.
///
/// Shutdown is deliberately strict: nOS-V may only be finalized on the owner
/// thread and never through an inherited post-fork descriptor graph.
#[derive(Debug)]
#[non_exhaustive]
pub enum ShutdownError {
    /// Shutdown must run on the creating thread.
    WrongThread,
    /// The runtime was inherited across `fork`.
    ForkedProcess,
    /// Native teardown failed.
    Native(NativeError),
}

impl fmt::Display for ShutdownError {
    /// Formats the error without calling back into nOS-V.
    ///
    /// Keeping formatting purely in Rust makes errors safe to display after the
    /// runtime has shut down or in a forked child.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongThread => f.write_str("runtime shutdown must run on its owner thread"),
            Self::ForkedProcess => f.write_str("runtime was inherited across fork"),
            Self::Native(error) => write!(f, "runtime shutdown failed: {error}"),
        }
    }
}

impl Error for ShutdownError {}

/// The unsuccessful terminal result of a spawned task.
///
/// A [`crate::JoinHandle`] publishes this error only after the Rust future has
/// been dropped and the corresponding native descriptor has been destroyed.
#[derive(Debug)]
#[non_exhaustive]
pub enum JoinError {
    /// Cancellation won its race with completion.
    Cancelled,
    /// The future panicked while being polled or dropped.
    Panic(Box<dyn Any + Send + 'static>),
    /// An internal native operation failed.
    Runtime(NativeError),
}

impl JoinError {
    /// Reports whether cooperative cancellation won the terminal race.
    ///
    /// Cancellation is not preemptive: this can become true only after the
    /// future returns control from `poll` and the executor observes the request.
    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }
    /// Reports whether polling or destroying the task produced a panic.
    ///
    /// This is always false under an aborting panic profile because such a panic
    /// terminates the process before a `JoinError` can be created.
    pub fn is_panic(&self) -> bool {
        matches!(self, Self::Panic(_))
    }
    /// Extracts the captured panic payload, consuming this error.
    ///
    /// Cancellation and native-runtime failures return `None`. The payload can be
    /// passed to [`std::panic::resume_unwind`] when join semantics should mirror
    /// an ordinary thread join.
    pub fn into_panic(self) -> Option<Box<dyn Any + Send + 'static>> {
        match self {
            Self::Panic(payload) => Some(payload),
            _ => None,
        }
    }
}

impl fmt::Display for JoinError {
    /// Formats the error without calling back into nOS-V.
    ///
    /// Keeping formatting purely in Rust makes errors safe to display after the
    /// runtime has shut down or in a forked child.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => f.write_str("task was cancelled"),
            Self::Panic(_) => f.write_str("task panicked"),
            Self::Runtime(error) => write!(f, "task failed in the native runtime: {error}"),
        }
    }
}

impl Error for JoinError {}
