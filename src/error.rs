//! Errors reported by the safe runtime.

use std::{any::Any, error::Error, fmt};

/// A native nOS-V error code.
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
    /// An nOS-V task context is required.
    OutsideTask,
    /// A native resource is busy.
    Busy,
    /// A timed native operation expired.
    Timeout,
    /// A code unknown to this crate version.
    Unknown(i32),
}

impl NativeError {
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

    /// Returns the underlying integer code.
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidCallback => "invalid callback",
            Self::InvalidMetadataSize => "invalid metadata size",
            Self::InvalidOperation => "invalid native operation",
            Self::InvalidParameter => "invalid parameter",
            Self::NotInitialized => "nOS-V is not initialized",
            Self::OutOfMemory => "native allocation failed",
            Self::OutsideTask => "operation requires an nOS-V task",
            Self::Busy => "native resource is busy",
            Self::Timeout => "native operation timed out",
            Self::Unknown(_) => "unknown native error",
        };
        write!(f, "{message} (code {})", self.code())
    }
}

impl Error for NativeError {}

/// Failure while constructing a runtime.
#[derive(Debug)]
#[non_exhaustive]
pub enum InitError {
    /// Runtime construction was attempted from inside a nOS-V task.
    AlreadyInTask,
    /// nOS-V rejected initialization.
    Native(NativeError),
}

impl fmt::Display for InitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyInTask => f.write_str("cannot create a runtime inside an nOS-V task"),
            Self::Native(error) => write!(f, "runtime initialization failed: {error}"),
        }
    }
}

impl Error for InitError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Native(error) => Some(error),
            _ => None,
        }
    }
}

impl From<NativeError> for InitError {
    fn from(value: NativeError) -> Self {
        Self::Native(value)
    }
}

/// The runtime is no longer accepting operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeClosed;

impl fmt::Display for RuntimeClosed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the nOS-V runtime is closed")
    }
}

impl Error for RuntimeClosed {}

/// Failure to create a spawned future.
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

impl fmt::Display for SpawnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RuntimeClosed => RuntimeClosed::fmt(&RuntimeClosed, f),
            Self::ForkedProcess => f.write_str("runtime handle was inherited across fork"),
            Self::Native(error) => write!(f, "could not spawn task: {error}"),
        }
    }
}

impl Error for SpawnError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Native(error) => Some(error),
            _ => None,
        }
    }
}

impl From<NativeError> for SpawnError {
    fn from(value: NativeError) -> Self {
        Self::Native(value)
    }
}

/// Failure before a root future could be driven.
#[derive(Debug)]
#[non_exhaustive]
pub enum BlockOnError {
    /// Only the creating thread may attach.
    WrongThread,
    /// `block_on` calls may not be nested.
    Nested,
    /// The caller is already an nOS-V task.
    AlreadyInTask,
    /// The runtime was inherited across `fork`.
    ForkedProcess,
    /// The runtime is closing or closed.
    RuntimeClosed,
    /// A native operation failed.
    Native(NativeError),
}

impl fmt::Display for BlockOnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongThread => f.write_str("block_on must run on the runtime owner thread"),
            Self::Nested => f.write_str("nested block_on is not supported"),
            Self::AlreadyInTask => f.write_str("block_on cannot run inside an nOS-V task"),
            Self::ForkedProcess => f.write_str("runtime was inherited across fork"),
            Self::RuntimeClosed => RuntimeClosed::fmt(&RuntimeClosed, f),
            Self::Native(error) => write!(f, "block_on native operation failed: {error}"),
        }
    }
}

impl Error for BlockOnError {}

/// Failure while shutting a runtime down.
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongThread => f.write_str("runtime shutdown must run on its owner thread"),
            Self::ForkedProcess => f.write_str("runtime was inherited across fork"),
            Self::Native(error) => write!(f, "runtime shutdown failed: {error}"),
        }
    }
}

impl Error for ShutdownError {}

/// The terminal result of a spawned task.
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
    /// Returns true for cooperative cancellation.
    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }
    /// Returns true when the task captured a panic.
    pub fn is_panic(&self) -> bool {
        matches!(self, Self::Panic(_))
    }
    /// Returns the captured payload, consuming this error.
    pub fn into_panic(self) -> Option<Box<dyn Any + Send + 'static>> {
        match self {
            Self::Panic(payload) => Some(payload),
            _ => None,
        }
    }
}

impl fmt::Display for JoinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => f.write_str("task was cancelled"),
            Self::Panic(_) => f.write_str("task panicked"),
            Self::Runtime(error) => write!(f, "task failed in the native runtime: {error}"),
        }
    }
}

impl Error for JoinError {}
