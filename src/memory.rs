//! Shared-memory statistics exposed by nOS-V.

use crate::error::NativeError;
use std::mem::MaybeUninit;

/// A checked snapshot of nOS-V shared-memory usage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MemoryStats {
    /// Bytes currently in use.
    pub used: usize,
    /// Total shared-memory bytes.
    pub size: usize,
    /// Pressure in `0.0..=1.0`.
    pub pressure: f32,
}

impl MemoryStats {
    pub(crate) fn query() -> Result<Self, NativeError> {
        let (mut used, mut size, mut pressure) = (
            MaybeUninit::<usize>::uninit(),
            MaybeUninit::<usize>::uninit(),
            MaybeUninit::<f32>::uninit(),
        );
        NativeError::from_code({
            /* SAFETY: writable output. */
            unsafe { nosv_sys::nosv_memory_get_used(used.as_mut_ptr()) }
        })?;
        NativeError::from_code({
            /* SAFETY: writable output. */
            unsafe { nosv_sys::nosv_memory_get_size(size.as_mut_ptr()) }
        })?;
        NativeError::from_code({
            /* SAFETY: writable output. */
            unsafe { nosv_sys::nosv_memory_get_pressure(pressure.as_mut_ptr()) }
        })?;
        // SAFETY: the successful calls initialized each output.
        let (used, size, pressure) = unsafe {
            (
                used.assume_init(),
                size.assume_init(),
                pressure.assume_init(),
            )
        };
        if used > size || !pressure.is_finite() || !(0.0..=1.0).contains(&pressure) {
            return Err(NativeError::InvalidOperation);
        }
        Ok(Self {
            used,
            size,
            pressure,
        })
    }
}
