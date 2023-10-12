use std::result;

mod error;
pub mod linalg;

#[cfg(feature = "gpu_accel")]
pub mod gpu;

pub use error::Jeeperr;

pub type Result<T> = result::Result<T, Jeeperr>;