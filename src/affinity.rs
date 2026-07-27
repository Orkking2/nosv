//! Validated task affinity values.

use crate::{
    error::NativeError,
    topology::{CpuId, NumaNodeId},
};

const MAX_INDEX: u32 = (1 << 29) - 1;

/// Whether placement is preferred or mandatory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AffinityKind {
    /// Prefer the target.
    Preferred,
    /// Require the target.
    Strict,
}

/// A validated nOS-V placement target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AffinityTarget {
    /// A system CPU.
    Cpu(CpuId),
    /// A system NUMA node.
    NumaNode(NumaNodeId),
    /// A native complex-set identifier.
    ComplexSet(u32),
}

/// Placement fixed before a task's first submission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Affinity {
    /// Leave placement to nOS-V.
    Any,
    /// Place at `target` with the requested strength.
    Target {
        /// Placement strength.
        kind: AffinityKind,
        /// Placement target.
        target: AffinityTarget,
    },
}

impl Affinity {
    /// Prefers a CPU.
    pub const fn preferred_cpu(cpu: CpuId) -> Self {
        Self::Target {
            kind: AffinityKind::Preferred,
            target: AffinityTarget::Cpu(cpu),
        }
    }
    /// Requires a CPU.
    pub const fn strict_cpu(cpu: CpuId) -> Self {
        Self::Target {
            kind: AffinityKind::Strict,
            target: AffinityTarget::Cpu(cpu),
        }
    }
    /// Prefers a NUMA node.
    pub const fn preferred_numa_node(node: NumaNodeId) -> Self {
        Self::Target {
            kind: AffinityKind::Preferred,
            target: AffinityTarget::NumaNode(node),
        }
    }
    /// Requires a NUMA node.
    pub const fn strict_numa_node(node: NumaNodeId) -> Self {
        Self::Target {
            kind: AffinityKind::Strict,
            target: AffinityTarget::NumaNode(node),
        }
    }

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
