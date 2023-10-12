use std::result;

mod error;
pub mod geo;
pub mod linalg;

#[cfg(feature = "gpu_accel")]
pub mod gpu;

#[cfg(feature = "gpu_accel")]
pub mod geogpu;

pub use error::Jeeperr;

pub type Result<T> = result::Result<T, Jeeperr>;