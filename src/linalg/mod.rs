//! Module containing the Matrix struct and associated utilities.

mod matrix;
mod utils;
mod bool_ops;
mod ref_ops;

mod sparse_matrix;

pub use matrix::Matrix;
pub use utils::Axis;
pub use utils::MatrixUtilities;
pub use bool_ops::BooleanMatrixOperations;
pub use ref_ops::ReferenceOperations;

pub use sparse_matrix::SparseMatrix;