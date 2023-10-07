use std::result;

mod error;
mod linalg;
mod gpu;

pub use error::Jeeperr;
pub use linalg::Matrix;
pub use gpu::Calculator;

pub type Result<T> = result::Result<T, Jeeperr>;