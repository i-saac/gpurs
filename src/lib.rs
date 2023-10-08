use std::result;

mod error;
mod linalg;
pub mod gpu;

pub use error::Jeeperr;
pub use linalg::Matrix;
pub use linalg::utils;
pub use gpu::Calculator;

pub type Result<T> = result::Result<T, Jeeperr>;