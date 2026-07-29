//! Safe, futures-based access to the nOS-V task runtime.
//!
//! The crate deliberately keeps nOS-V descriptors private. Spawned futures are
//! `Send + 'static` because a nOS-v task may resume on another worker pthread;
//! a root future passed to [`Runtime::block_on`] may borrow local state and need
//! not be `Send` because it remains on the attached calling thread.
//!
//! Scheduling is cooperative. A future that never returns from `poll` cannot be
//! cancelled and can prevent runtime shutdown. Native configuration is process
//! global and installations may enable FTZ/DAZ floating-point modes.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

mod ffi;
mod util;

pub mod affinity;
pub mod error;
#[cfg(feature = "io-uring")]
pub mod io_uring;
pub mod memory;
pub mod runtime;
pub mod task;
#[cfg(feature = "time")]
pub mod time;
pub mod topology;

pub use affinity::{Affinity, AffinityKind, AffinityTarget};
pub use error::{
    BlockOnError, InitError, JoinError, NativeError, RuntimeClosed, ShutdownError, SpawnError,
};
pub use memory::MemoryStats;
pub use runtime::{Handle, Runtime, RuntimeBuilder};
pub use task::{AbortHandle, JoinHandle, TaskBuilder};
pub use topology::{CpuId, DomainId, NumaNodeId, Topology, TopologyLevel};

/// Creates a task on the runtime currently polling this future.
///
/// This crate-root convenience function is equivalent to [`task::spawn`]. A current runtime exists
/// only while [`Runtime::block_on`] is polling its root future or a nOS-V task callback is polling a
/// spawned future. Code that already owns a [`Handle`] should use [`Handle::spawn`] explicitly.
///
/// Dropping the returned [`JoinHandle`] detaches the task; call [`JoinHandle::abort`] to request
/// cooperative cancellation.
///
/// # Errors
///
/// Returns [`SpawnError::RuntimeClosed`] outside a current runtime context, or forwards validation,
/// fork-safety, lifecycle, and native creation errors from the runtime.
pub fn spawn<F, T>(future: F) -> Result<JoinHandle<T>, SpawnError>
where
    F: std::future::Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    task::spawn(future)
}

#[cfg(test)]
/// Compile-time assertions for the runtime's thread-safety boundary.
mod trait_tests {
    use super::*;
    use static_assertions::{assert_impl_all, assert_not_impl_any};

    assert_not_impl_any!(Runtime: Send, Sync);
    assert_impl_all!(Handle: Send, Sync, Clone);
    assert_impl_all!(JoinHandle<u32>: Send, Sync);
    assert_impl_all!(AbortHandle: Send, Sync, Clone);
}
