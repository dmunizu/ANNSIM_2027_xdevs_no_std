#![no_std]
#![forbid(unsafe_code)]
//! DEVS models for the ANNSIM 2027 uBLOOM demonstration.
//!
//! The xdevs backend is chosen by feature so the same models build for both
//! target families:
//!
//! The models themselves are backend-agnostic, and the binary crate that runs the simulation picks the
//! backend (Cargo unifies features across a build graph).

pub mod alert;
pub mod forecast;
pub mod system;

pub use system::ForecastSystem;
