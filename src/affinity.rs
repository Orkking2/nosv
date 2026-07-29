//! Validated task affinity values.
//!
//! nOS-V encodes affinity as C bitfields. This module replaces those layout-
//! sensitive fields with ordinary Rust enums, validates the 29-bit native index,
//! and converts back only while an unsubmitted task is being built.

use crate::{
    error::NativeError,
    topology::{CpuId, NumaNodeId},
};

/// Largest index representable by the native affinity bitfield.
///
/// The public CPU and NUMA identifiers also participate in non-affinity APIs and
/// may therefore be wider; conversion rejects them only when affinity encoding
/// actually requires this narrower representation.
const MAX_INDEX: u32 = (1 << 29) - 1;

/// Whether placement is preferred or mandatory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AffinityKind {
    /// Prefer the target but allow nOS-V to choose another placement.
    Preferred,
    /// Require nOS-V to place the task at the selected target.
    Strict,
}

/// A validated nOS-V placement target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AffinityTarget {
    /// A system CPU identifier from the runtime's topology view.
    Cpu(CpuId),
    /// A system NUMA-node identifier from the runtime's topology view.
    NumaNode(NumaNodeId),
    /// A native user-defined complex-set identifier.
    ComplexSet(u32),
}

/// Placement fixed before a task's first native submission.
///
/// Affinity is deliberately immutable after [`crate::task::TaskBuilder::spawn`]
/// because nOS-V documents concurrent or post-submit mutation as undefined.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Affinity {
    /// Leave placement entirely to nOS-V's scheduler.
    Any,
    /// Apply `target` as either a preference or strict constraint.
    Target {
        /// Placement strength.
        kind: AffinityKind,
        /// Placement target.
        target: AffinityTarget,
    },
}

impl Affinity {
    /// Constructs a soft preference for `cpu`.
    ///
    /// The CPU index is validated against the C bitfield when the task is
    /// spawned, so this constant constructor remains infallible and ergonomic.
    pub const fn preferred_cpu(cpu: CpuId) -> Self {
        Self::Target {
            kind: AffinityKind::Preferred,
            target: AffinityTarget::Cpu(cpu),
        }
    }

    /// Constructs a strict placement requirement for `cpu`.
    ///
    /// Spawn fails with [`NativeError::InvalidParameter`] if the identifier does
    /// not fit the native affinity representation.
    pub const fn strict_cpu(cpu: CpuId) -> Self {
        Self::Target {
            kind: AffinityKind::Strict,
            target: AffinityTarget::Cpu(cpu),
        }
    }

    /// Constructs a soft preference for `node`.
    ///
    /// This selects a system NUMA identifier rather tha nOS-v's logical index,
    /// matching the values returned by [`crate::Topology::numa_nodes`].
    pub const fn preferred_numa_node(node: NumaNodeId) -> Self {
        Self::Target {
            kind: AffinityKind::Preferred,
            target: AffinityTarget::NumaNode(node),
        }
    }

    /// Constructs a strict placement requirement for `node`.
    ///
    /// Validation is deferred to task construction because the same ID wrapper
    /// can be used by wider topology queries that do not have a 29-bit limit.
    pub const fn strict_numa_node(node: NumaNodeId) -> Self {
        Self::Target {
            kind: AffinityKind::Strict,
            target: AffinityTarget::NumaNode(node),
        }
    }

    /// Encodes this safe value into nOS-V's generated bitfield structure.
    ///
    /// This is kept crate-private so generated setters and ABI types never
    /// become part of the safe public interface. The method explicitly fills
    /// every field and rejects indices that would otherwise be truncated.
    pub(crate) fn to_raw(self) -> Result<nosv_sys::nosv_affinity_t, NativeError> {
        let mut raw = nosv_sys::nosv_affinity_t::default();
        match self {
            Self::Any => {
                raw.set_level(nosv_sys::NOSV_AFFINITY_LEVEL_NONE);
                raw.set_type(nosv_sys::NOSV_AFFINITY_TYPE_PREFERRED);
                raw.set_index(0);
            }
            Self::Target { kind, target } => {
                let (level, index) = match target {
                    AffinityTarget::Cpu(id) => (nosv_sys::NOSV_AFFINITY_LEVEL_CPU, id.get()),
                    AffinityTarget::NumaNode(id) => (nosv_sys::NOSV_AFFINITY_LEVEL_NUMA, id.get()),
                    AffinityTarget::ComplexSet(id) => {
                        (nosv_sys::NOSV_AFFINITY_LEVEL_USER_COMPLEX, id)
                    }
                };
                if index > MAX_INDEX {
                    return Err(NativeError::InvalidParameter);
                }
                raw.set_level(level);
                raw.set_type(match kind {
                    AffinityKind::Preferred => nosv_sys::NOSV_AFFINITY_TYPE_PREFERRED,
                    AffinityKind::Strict => nosv_sys::NOSV_AFFINITY_TYPE_STRICT,
                });
                raw.set_index(index);
            }
        }
        Ok(raw)
    }
}

#[cfg(test)]
#[allow(clippy::missing_docs_in_private_items)]
mod tests {
    use super::*;

    #[test]
    fn affinity_index_boundary_is_checked_without_truncation() {
        let maximum = CpuId::new(MAX_INDEX).unwrap();
        assert!(Affinity::strict_cpu(maximum).to_raw().is_ok());
        let too_large = CpuId::new(MAX_INDEX + 1).unwrap();
        assert!(matches!(
            Affinity::preferred_cpu(too_large).to_raw(),
            Err(NativeError::InvalidParameter)
        ));
        assert!(Affinity::Any.to_raw().is_ok());
    }
}
