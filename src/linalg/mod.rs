//! Module containing the Matrix struct and associated utilities.

mod matrix;
mod utils;
mod bool_ops;
mod matrix_ops;
mod matrix_ref_ops;

mod sparse_matrix;
mod sparse_ref_ops;

pub use matrix::Matrix;
pub use utils::Axis;
pub use utils::MatrixUtilities;
pub use bool_ops::BooleanMatrixOperations;

pub use sparse_matrix::SparseMatrix;
pub use sparse_ref_ops::SparseReferenceOperations;