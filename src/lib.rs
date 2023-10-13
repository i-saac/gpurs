use std::result;

mod error;
pub mod geo;
pub mod linalg;

#[cfg(feature = "gpu_accel")]
pub mod gpu;

#[cfg(feature = "gpu_accel")]
pub mod geogpu;

pub use error::Jeeperr;

/// Shorthand result type for gpurs.
pub type Result<T> = result::Result<T, Jeeperr>;

/// Trait allowing objects to be implemented for single or double precision floats
pub trait IsFloat {}
impl IsFloat for f32 {}
impl IsFloat for f64 {}