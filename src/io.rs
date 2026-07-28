//! Audited configuration boundary reserved for the v0.2 io_uring driver.
//!
//! Enabling this feature does not yet submit kernel operations. It fixes and
//! tests the public configuration and private generation-token representation
//! before buffer ownership and cancellation are introduced.

use std::{error::Error, fmt, time::Duration};

/// Configuration for the planned single-owner io_uring driver.
///
/// The current feature establishes the validated configuration and generation-token boundary; it
/// does not yet submit kernel operations. The eventual driver will own the ring from one nOS-V task
/// and use these limits to bound work performed per cooperative scheduler pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IoUringConfig {
    /// Submission/completion ring entry count.
    pub entries: u32,
    /// Maximum commands and completions drained per scheduler pass.
    pub completion_batch: usize,
    /// Maximum initial delay before polling the completion queue again.
    pub poll_interval: Duration,
}

impl Default for IoUringConfig {
    /// Returns a conservative initial queue depth, batch size, and 50-microsecond polling interval.
    ///
    /// These are starting values rather than performance guarantees; applications should benchmark
    /// the completed driver against their workload and completion-latency requirements.
    fn default() -> Self {
        Self {
            entries: 256,
            completion_batch: 256,
            poll_interval: Duration::from_micros(50),
        }
    }
}

impl IoUringConfig {
    /// Validates values before any ring or native driver task is created.
    ///
    /// Requiring all progress-related values to be nonzero prevents constructing a driver that can
    /// never accept work, drain a completion, or revisit an in-flight operation.
    ///
    /// # Errors
    ///
    /// Returns the variant of [`InvalidIoUringConfig`] corresponding to the first zero field.
    pub fn validate(self) -> Result<Self, InvalidIoUringConfig> {
        if self.entries == 0 {
            return Err(InvalidIoUringConfig::ZeroEntries);
        }
        if self.completion_batch == 0 {
            return Err(InvalidIoUringConfig::ZeroCompletionBatch);
        }
        if self.poll_interval.is_zero() {
            return Err(InvalidIoUringConfig::ZeroPollInterval);
        }
        Ok(self)
    }
}

/// Invalid io_uring driver configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum InvalidIoUringConfig {
    /// The kernel ring cannot contain zero entries.
    ZeroEntries,
    /// A zero batch would prevent progress.
    ZeroCompletionBatch,
    /// The initial polling driver requires a positive interval.
    ZeroPollInterval,
}

impl fmt::Display for InvalidIoUringConfig {
    /// Describes the field that would prevent driver progress.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroEntries => f.write_str("io_uring entries must be non-zero"),
            Self::ZeroCompletionBatch => f.write_str("io_uring completion batch must be non-zero"),
            Self::ZeroPollInterval => f.write_str("io_uring poll interval must be non-zero"),
        }
    }
}

impl Error for InvalidIoUringConfig {}

/// Generation-tagged identity reserved for an io_uring operation slab entry.
///
/// The kernel copies SQE `user_data` unchanged into its CQE. Encoding an integer token instead of a
/// Rust pointer lets the driver validate slot reuse and prevents the kernel-facing ABI from
/// borrowing a task, waker, buffer, or native descriptor.
#[allow(dead_code)] // First production consumer: the v0.2 operation slab.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OperationToken {
    /// Slab index owning the operation state.
    slot: u32,
    /// Generation distinguishing reuse of the same slab index.
    generation: u32,
}

#[allow(dead_code)] // First production consumer: the v0.2 operation slab.
impl OperationToken {
    /// Constructs a token from its slab index and current generation.
    pub(crate) const fn new(slot: u32, generation: u32) -> Self {
        Self { slot, generation }
    }

    /// Packs generation into the high half and slot into the low half of `user_data`.
    pub(crate) const fn encode(self) -> u64 {
        (self.generation as u64) << 32 | self.slot as u64
    }

    /// Recovers both token halves from a completion's `user_data` value.
    ///
    /// Decoding alone will not authorize access: the future operation slab must also compare the
    /// decoded generation with the live entry before resolving a CQE.
    pub(crate) const fn decode(encoded: u64) -> Self {
        Self {
            slot: encoded as u32,
            generation: (encoded >> 32) as u32,
        }
    }
}

#[cfg(test)]
/// Unit tests for the kernel-facing token and configuration invariants.
mod tests {
    use super::*;

    /// Verifies that neither 32-bit half is truncated or swapped by token encoding.
    #[test]
    fn token_round_trip_preserves_both_halves() {
        let token = OperationToken::new(0xfedc_ba98, 0x7654_3210);
        assert_eq!(OperationToken::decode(token.encode()), token);
    }

    /// Keeps default values synchronized with all progress requirements in `validate`.
    #[test]
    fn defaults_are_valid() {
        assert_eq!(
            IoUringConfig::default().validate(),
            Ok(IoUringConfig::default())
        );
    }
}
