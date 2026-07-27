//! Safe, futures-based access to the nOS-V task runtime.
//!
//! The crate deliberately keeps nOS-V descriptors private. Spawned futures are
//! `Send + 'static` because an nOS-V task may resume on another worker pthread;
//! a root future passed to [`Runtime::block_on`] may borrow local state and need
//! not be `Send` because it remains on the attached calling thread.
//!
//! Scheduling is cooperative. A future that never returns from `poll` cannot be
//! cancelled and can prevent runtime shutdown. Native configuration is process
//! global and installations may enable FTZ/DAZ floating-point modes.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

mod ffi;

pub mod affinity;
pub mod error;
#[cfg(feature = "io-uring")]
pub mod io;
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
pub fn spawn<F, T>(future: F) -> Result<JoinHandle<T>, SpawnError>
where
    F: std::future::Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    task::spawn(future)
}

#[cfg(test)]
mod trait_tests {
    use super::*;
    use static_assertions::{assert_impl_all, assert_not_impl_any};

    assert_not_impl_any!(Runtime: Send, Sync);
    assert_impl_all!(Handle: Send, Sync, Clone);
    assert_impl_all!(JoinHandle<u32>: Send, Sync);
    assert_impl_all!(AbortHandle: Send, Sync, Clone);
}
