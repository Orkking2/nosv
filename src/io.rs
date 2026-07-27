//! Audited configuration boundary reserved for the v0.2 io_uring driver.
//!
//! Enabling this feature does not yet submit kernel operations. It fixes and
//! tests the public configuration and private generation-token representation
//! before buffer ownership and cancellation are introduced.

use std::{error::Error, fmt, time::Duration};

/// Configuration for the future single-owner io_uring driver.
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroEntries => f.write_str("io_uring entries must be non-zero"),
            Self::ZeroCompletionBatch => f.write_str("io_uring completion batch must be non-zero"),
            Self::ZeroPollInterval => f.write_str("io_uring poll interval must be non-zero"),
        }
    }
}

impl Error for InvalidIoUringConfig {}

#[allow(dead_code)] // First production consumer: the v0.2 operation slab.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OperationToken {
    slot: u32,
    generation: u32,
}

#[allow(dead_code)] // First production consumer: the v0.2 operation slab.
impl OperationToken {
    pub(crate) const fn new(slot: u32, generation: u32) -> Self {
        Self { slot, generation }
    }

    pub(crate) const fn encode(self) -> u64 {
        (self.generation as u64) << 32 | self.slot as u64
    }

    pub(crate) const fn decode(encoded: u64) -> Self {
        Self {
            slot: encoded as u32,
            generation: (encoded >> 32) as u32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_round_trip_preserves_both_halves() {
        let token = OperationToken::new(0xfedc_ba98, 0x7654_3210);
        assert_eq!(OperationToken::decode(token.encode()), token);
    }

    #[test]
    fn defaults_are_valid() {
        assert_eq!(
            IoUringConfig::default().validate(),
            Ok(IoUringConfig::default())
        );
    }
}
