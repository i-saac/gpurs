use crate::IsFloat;
use crate::Result;
use crate::Jeeperr;

use crate::linalg::Matrix;
use crate::linalg::SparseMatrix;

pub trait SparseReferenceOperations<T: IsFloat + std::fmt::Debug + Copy + Clone> {
    fn ref_mul(left: &SparseMatrix<T>, right: &Matrix<T>) -> Result<Matrix<T>>;
}

impl SparseReferenceOperations<f64> for SparseMatrix<f64> {
    fn ref_mul(left: &SparseMatrix<f64>, right: &Matrix<f64>) -> Result<Matrix<f64>> {
        if left.get_cols() != right.get_rows() {
            return Err(Jeeperr::DimensionError)
        }

        let new_n_elements: usize = left.get_rows() * right.get_cols();
        let mut output_data: Vec<f64> = vec![0.0; new_n_elements];

        for (row_idx, row_map) in left.get_data().iter().enumerate() {
            for col in 0..right.get_cols() {
                output_data[row_idx * right.get_cols() + col] = row_map.iter()
                    .map(|(&lhs_col, lhs_val)| lhs_val * right[[lhs_col, col]])
                    .sum();
            }
        }

        return Ok(Matrix::new(output_data, left.get_rows(), right.get_cols())?)
    }
}