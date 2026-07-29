//! Safe copies of native topology information.

use crate::{
    error::NativeError,
    runtime::{Lifecycle, RuntimeCore},
};
use std::{ptr::NonNull, sync::Weak};

/// Defines a non-negative topology identifier with checked C-ABI conversion.
///
/// CPU and NUMA identifiers have identical representation rules but remain distinct Rust types so
/// they cannot be mixed accidentally at call sites.
macro_rules! id_type {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(
            /// Non-negative system identifier represented by nOS-V's signed C API.
            u32,
        );
        impl $name {
            /// Constructs an identifier accepted by the signed native API.
            ///
            /// nOS-V represents topology identifiers as `i32`; rejecting larger Rust values here
            /// ensures every stored identifier can be passed across FFI without truncation.
            ///
            /// # Errors
            ///
            /// Returns [`NativeError::InvalidParameter`] when `value` exceeds `i32::MAX`.
            pub fn new(value: u32) -> Result<Self, NativeError> {
                if value > i32::MAX as u32 {
                    Err(NativeError::InvalidParameter)
                } else {
                    Ok(Self(value))
                }
            }
            /// Returns the non-negative system identifier.
            pub const fn get(self) -> u32 {
                self.0
            }
            /// Converts a native signed identifier while rejecting negative sentinel values.
            pub(crate) fn from_native(value: i32) -> Result<Self, NativeError> {
                u32::try_from(value)
                    .map(Self)
                    .map_err(|_| NativeError::InvalidParameter)
            }
        }
    };
}
id_type!(CpuId, "A system CPU identifier.");
id_type!(NumaNodeId, "A system NUMA-node identifier.");

/// A topology level exposed by nOS-V.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum TopologyLevel {
    /// A machine node.
    Node,
    /// A NUMA node.
    Numa,
    /// A complex set.
    ComplexSet,
    /// A physical core.
    Core,
    /// A CPU.
    Cpu,
}

impl TopologyLevel {
    /// Converts this checked Rust level to the corresponding nOS-V enumeration value.
    pub(crate) const fn raw(self) -> nosv_sys::nosv_topo_level_t {
        match self {
            Self::Node => nosv_sys::NOSV_TOPO_LEVEL_NODE,
            Self::Numa => nosv_sys::NOSV_TOPO_LEVEL_NUMA,
            Self::ComplexSet => nosv_sys::NOSV_TOPO_LEVEL_COMPLEX_SET,
            Self::Core => nosv_sys::NOSV_TOPO_LEVEL_CORE,
            Self::Cpu => nosv_sys::NOSV_TOPO_LEVEL_CPU,
        }
    }
}

/// A topology domain at a particular level.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DomainId {
    /// Topology hierarchy in which `system_id` is meaningful.
    level: TopologyLevel,
    /// Non-negative system identifier used by nOS-V queries.
    system_id: i32,
}
impl DomainId {
    /// Constructs a domain identifier at the supplied topology level.
    ///
    /// # Errors
    ///
    /// Returns [`NativeError::InvalidParameter`] for a negative native identifier.
    pub fn new(level: TopologyLevel, system_id: i32) -> Result<Self, NativeError> {
        if system_id < 0 {
            Err(NativeError::InvalidParameter)
        } else {
            Ok(Self { level, system_id })
        }
    }
    /// Returns the topology level in which this identifier is defined.
    pub const fn level(self) -> TopologyLevel {
        self.level
    }
    /// Returns the non-negative system identifier expected by nOS-V.
    pub const fn system_id(self) -> i32 {
        self.system_id
    }
}

/// Read-only topology query capability tied to a live runtime.
///
/// Each query upgrades the weak reference, verifies the process identity, and holds the runtime
/// lifecycle lock across FFI. A topology value therefore does not keep a runtime alive and becomes
/// harmless after shutdown or in a child process created by `fork`.
#[derive(Clone)]
pub struct Topology {
    /// Runtime whose initialized native topology view is queried.
    pub(crate) runtime: Weak<RuntimeCore>,
}
impl Topology {
    /// Runs a topology query while pinning the runtime in its running lifecycle state.
    ///
    /// Holding the state mutex through `query` prevents owner-thread shutdown from calling
    /// `nosv_shutdown` while the native topology allocation is being inspected and freed.
    fn with_live<R>(
        &self,
        query: impl FnOnce() -> Result<R, NativeError>,
    ) -> Result<R, NativeError> {
        let runtime = self.runtime.upgrade().ok_or(NativeError::NotInitialized)?;
        if !runtime.process_matches() {
            return Err(NativeError::NotInitialized);
        }
        let state = runtime.lock_state();
        if state.lifecycle != Lifecycle::Running {
            return Err(NativeError::NotInitialized);
        }
        query()
    }

    /// Returns a Rust-owned snapshot of visible system CPUs.
    ///
    /// The malloc-owned native array is copied and released before this method returns.
    pub fn cpus(&self) -> Result<Vec<CpuId>, NativeError> {
        self.with_live(|| {
            // SAFETY: lifecycle is locked in Running until this query returns.
            let count = unsafe { nosv_sys::nosv_get_num_cpus() };
            if count < 0 {
                return native_count_error(count);
            }
            // SAFETY: the malloc-owned result is copied and freed by copy_ids.
            let values = unsafe { nosv_sys::nosv_get_available_cpus() };
            copy_ids(count, values)?
                .into_iter()
                .map(CpuId::from_native)
                .collect()
        })
    }

    /// Returns a Rust-owned snapshot of visible system NUMA nodes.
    ///
    /// Negative native counts and identifiers are treated as errors rather than cast to unsigned
    /// Rust values.
    pub fn numa_nodes(&self) -> Result<Vec<NumaNodeId>, NativeError> {
        self.with_live(|| {
            // SAFETY: lifecycle is locked in Running until this query returns.
            let count = unsafe { nosv_sys::nosv_get_num_numa_nodes() };
            if count < 0 {
                return native_count_error(count);
            }
            // SAFETY: the malloc-owned result is copied and freed by copy_ids.
            let values = unsafe { nosv_sys::nosv_get_available_numa_nodes() };
            copy_ids(count, values)?
                .into_iter()
                .map(NumaNodeId::from_native)
                .collect()
        })
    }

    /// Returns a Rust-owned snapshot of visible domains at `level`.
    ///
    /// Each returned [`DomainId`] retains its level because system identifiers are interpreted in
    /// the context of a particular topology hierarchy.
    pub fn domains(&self, level: TopologyLevel) -> Result<Vec<DomainId>, NativeError> {
        self.with_live(|| {
            // SAFETY: lifecycle is locked and level maps to a valid native enum.
            let count = unsafe { nosv_sys::nosv_get_num_domains(level.raw()) };
            if count < 0 {
                return native_count_error(count);
            }
            // SAFETY: the malloc-owned result is copied and freed by copy_ids.
            let values = unsafe { nosv_sys::nosv_get_available_domains(level.raw()) };
            copy_ids(count, values)?
                .into_iter()
                .map(|id| DomainId::new(level, id))
                .collect()
        })
    }

    /// Returns the visible CPUs contained by `domain`.
    ///
    /// The domain's level and validated non-negative system identifier are forwarded together to
    /// prevent accidentally querying an identifier in the wrong hierarchy.
    pub fn cpus_in(&self, domain: DomainId) -> Result<Vec<CpuId>, NativeError> {
        self.with_live(|| {
            // SAFETY: lifecycle is locked and domain construction validated id.
            let count = unsafe {
                nosv_sys::nosv_get_num_cpus_in_domain(domain.level.raw(), domain.system_id)
            };
            if count < 0 {
                return native_count_error(count);
            }
            // SAFETY: the malloc-owned result is copied and freed by copy_ids.
            let values = unsafe {
                nosv_sys::nosv_get_available_cpus_in_domain(domain.level.raw(), domain.system_id)
            };
            copy_ids(count, values)?
                .into_iter()
                .map(CpuId::from_native)
                .collect()
        })
    }
}

/// Converts a negative native count into a typed error result.
///
/// A non-error success code in the count position violates the API contract and is represented as
/// [`NativeError::InvalidOperation`] rather than inventing a count.
fn native_count_error<T>(count: i32) -> Result<T, NativeError> {
    match NativeError::from_code(count) {
        Err(error) => Err(error),
        Ok(()) => Err(NativeError::InvalidOperation),
    }
}

/// Copies and frees an array allocated by a nOS-v topology query.
///
/// This function never uses `Vec::from_raw_parts`: the array was allocated by C's allocator, so it
/// is copied into Rust-owned storage and released with [`libc::free`]. Null pointers, negative
/// counts, and slice-size overflow are validated before dereferencing.
fn copy_ids(count: i32, values: *mut i32) -> Result<Vec<i32>, NativeError> {
    if count < 0 {
        free(values);
        return match NativeError::from_code(count) {
            Err(error) => Err(error),
            Ok(()) => Err(NativeError::InvalidOperation),
        };
    }
    if count == 0 {
        free(values);
        return Ok(Vec::new());
    }
    let ptr = NonNull::new(values).ok_or(NativeError::OutOfMemory)?;
    let len = usize::try_from(count).map_err(|_| NativeError::InvalidParameter)?;
    if len > isize::MAX as usize / std::mem::size_of::<i32>() {
        free(values);
        return Err(NativeError::InvalidParameter);
    }
    // SAFETY: a successful query returned `count` initialized integers.
    let copied = unsafe { std::slice::from_raw_parts(ptr.as_ptr(), len) }.to_vec();
    free(values);
    Ok(copied)
}

/// Releases a nullable topology array through the allocator that created it.
fn free(pointer: *mut i32) {
    if !pointer.is_null() {
        /* SAFETY: topology arrays are malloc-owned. */
        unsafe { libc::free(pointer.cast()) };
    }
}

#[cfg(test)]
#[allow(clippy::missing_docs_in_private_items)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_reject_signed_abi_overflow_and_negative_domains() {
        assert_eq!(CpuId::new(i32::MAX as u32).unwrap().get(), i32::MAX as u32);
        assert_eq!(
            CpuId::new(i32::MAX as u32 + 1),
            Err(NativeError::InvalidParameter)
        );
        assert_eq!(
            NumaNodeId::from_native(-1),
            Err(NativeError::InvalidParameter)
        );
        assert_eq!(
            DomainId::new(TopologyLevel::Cpu, -1),
            Err(NativeError::InvalidParameter)
        );
        let domain = DomainId::new(TopologyLevel::Numa, 7).unwrap();
        assert_eq!(domain.level(), TopologyLevel::Numa);
        assert_eq!(domain.system_id(), 7);
    }
}
