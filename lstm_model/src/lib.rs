//! uBLOOM bloom-forecasting LSTM, reimplemented in integer Rust.
//!
//! This crate is the model only: no TensorFlow Lite, no C++, no heap, no
//! floating point beyond the four output values. It is `no_std` and has no
//! dependencies.

#![no_std]
#![forbid(unsafe_code)]

#[rustfmt::skip]
mod data;
mod fixed;
mod lstm;

pub use data::{IN_PER_STEP, OUT_DIM, TIMESTEPS};
pub use lstm::predict;

/// Number of input features expected by [`predict`].
pub const FEATURES: usize = TIMESTEPS * IN_PER_STEP;
