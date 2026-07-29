//! Small synchronization helpers shared by runtime subsystems.
//!
//! Native callbacks contain panics at their ABI boundaries, but a panic can
//! still poison a Rust mutex before it is caught. Runtime teardown must continue
//! to access that state so it can retire native descriptors and kernel-visible
//! pointers safely. This module centralizes that poison-recovery policy.

use std::sync::{Mutex, MutexGuard, PoisonError};

/// Locks a mutex and recovers its protected value if the mutex was poisoned.
///
/// Poisoning indicates that another thread unwound while holding the lock; it
/// does not make abandoning runtime-owned state safe. Callers use this helper for
/// state whose invariants are either restored before a panic can escape or whose
/// cleanup must proceed even after a contained callback panic.
///
/// This function otherwise has the same blocking and reentrancy behavior as
/// [`Mutex::lock`]. In particular, it can block and must not be used to lock a
/// mutex recursively from the same thread.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
