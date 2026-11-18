//! The integer primitives TFLM and esp-nn use, transcribed one for one.
//! Only the functions used in the uBLOOM model are included here.
//!
//! Each function names the C code it mirrors. The goal is
//! to be bit-identical. For that reason,
//! some of the functions are different from the "obvious" Rust
//! translation of the C code, which would be more idiomatic but not bit-identical.

use core::cmp::{max, min};

/// `esp_nn_sat_round_doubling_high_mul`, i.e. gemmlowp's
/// `SaturatingRoundingDoublingHighMul`.
#[inline]
pub fn sat_round_doubling_high_mul(a: i32, b: i32) -> i32 {
    let overflow = a == i32::MIN && b == i32::MIN;

    // Round before truncating.
    let nudge: i64 = if (a < 0) == (b < 0) {
        1 << 30
    } else {
        1 - (1 << 30)
    };
    let m = (a as i64) * (b as i64) + nudge;

    // For a negative product, adding 0x7FFF_FFFF makes the following shift
    // truncate toward zero.
    let to_add = if m < 0 { 0x7FFF_FFFF_i64 } else { 0 };
    let r = ((m + to_add) >> 31) as i32;

    if overflow { i32::MAX } else { r }
}

/// `esp_nn_div_by_power_of_two`, i.e. gemmlowp's `RoundingDivideByPOT`.
#[inline]
pub fn div_by_power_of_two(val: i32, exponent: u32) -> i32 {
    // Theoretically, exponent should always be in [0, 31] due to quantization.
    // So this guard may be unnecessary.
    let exponent = min(exponent, 31);
    if exponent == 0 {
        return val;
    }
    let mask = ((1u32 << exponent) - 1) as i32;
    let remainder = val & mask;
    let mut result = val >> exponent;
    let threshold = (mask >> 1) + ((result < 0) as i32);
    if remainder > threshold {
        result += 1;
    }
    result
}

/// `esp_nn_multiply_by_quantized_mult`, apply a multiplier whose fixed point
/// sits at bit 31, with a positive `shift` meaning "pre-shift the input left".
#[inline]
pub fn multiply_by_quantized_mult(x: i32, mult: i32, shift: i32) -> i32 {
    let left = max(shift, 0) as u32;
    let right = max(-shift, 0) as u32;
    // The C original shifts an int32 and lets the high bits fall off.
    let pre = sat_round_doubling_high_mul(x << left, mult);
    div_by_power_of_two(pre, right)
}
