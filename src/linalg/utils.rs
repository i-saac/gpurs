use crate::IsFloat;
use crate::Result;
use crate::Jeeperr;
use crate::linalg::Matrix;

/// Axis enum for use in utility functions.
#[derive(Debug, Clone, Copy)]
pub enum Axis {
    /// Perform action along row dimension
    Row,
    /// Perform action along col dimension
    Col,
}

/// Trait containing utility functions for the Matrix struct.
pub trait MatrixUtilities<T: IsFloat + std::fmt::Debug + Copy + Clone> {
    /// Perform partial LU decomposition of a matrix.
    /// This function returns a single matrix LU that is equivalent to L + U - I.
    /// This is designed primarily as a helper function for linear_solve_matrix.
    /// This function can return an error.
    fn lu_decomp(matrix: &Matrix<T>) -> Result<Matrix<T>>;
    /// Solve system of linear equations Ax=b.
    /// This will output x as a Matrix given A and b.
    /// This function can return an error.
    fn linear_solve_matrix(a_mat: &Matrix<T>, b_vec: &Matrix<T>) -> Result<Matrix<T>>;

    /// Find absolute maximum value of matrix.
    /// This will find and output the single maximum value across all elements in the matrix.
    fn max(matrix: &Matrix<T>) -> T;
    /// Find maximum value of matrix along an axis.
    /// This will find the maximum value of each row or column and output them as a matrix.
    fn axis_max(matrix: &Matrix<T>, axis: Axis) -> Matrix<T>;
    /// Find absolute minimum value of matrix.
    /// This will find and output the single minimum value across all elements in the matrix.
    fn min(matrix: &Matrix<T>) -> T;
    /// Find minimum value of matrix along an axis.
    /// This will find the minimum value of each row or column and output them as a matrix.
    fn axis_min(matrix: &Matrix<T>, axis: Axis) -> Matrix<T>;

    /// Find sum of all elements in a matrix.
    fn sum(matrix: &Matrix<T>) -> T;
    /// Sum elements of a matrix along rows or columns, returning a column or row matrix.
    fn axis_sum(matrix: &Matrix<T>, axis: Axis) -> Matrix<T>;

    /// Concatenate slice of matrices together along an axis.
    /// Think of this as "stacking" the matrices together either top to bottom (Axis::Row) or left to right (Axis::Col)
    fn concatenate(matrices: &[Matrix<T>], axis: Axis) -> Result<Matrix<T>>;
}

impl MatrixUtilities<f32> for Matrix<f32> {
    /// Perform partial LU decomposition of a matrix.
    /// This function returns a single matrix LU that is equivalent to L + U - I.
    /// This is designed primarily as a helper function for linear_solve_matrix.
    /// This function can return an error.
    fn lu_decomp(matrix: &Matrix<f32>) -> Result<Matrix<f32>> {
        if matrix.get_rows() != matrix.get_cols() {
            return Err(Jeeperr::DimensionError)
        }
        let dim_n: usize = matrix.get_rows();
    
        let mut lu_data: Vec<f32> = vec![0.0; dim_n * dim_n];
        for idx in 0..dim_n {
            for jdx in idx..dim_n {
                let sum: f32 = (0..idx)
                    .into_iter()
                    .map(|kdx| lu_data[idx * dim_n + kdx] * lu_data[kdx * dim_n + jdx])
                    .sum();
                lu_data[idx * dim_n + jdx] = matrix[[idx, jdx]] - sum;
            }
            for jdx in (idx + 1)..dim_n {
                let sum: f32 = (0..idx)
                    .into_iter()
                    .map(|kdx| lu_data[jdx * dim_n + kdx] * lu_data[kdx * dim_n + idx])
                    .sum();
                lu_data[jdx * dim_n + idx] = (matrix[[jdx, idx]] - sum) / lu_data[idx * dim_n + idx];
            }
        }
    
        return Ok(Matrix::new(lu_data, dim_n, dim_n)?)
    }

    /// Solve system of linear equations Ax=b.
    /// This will output x as a Matrix given A and b.
    /// This function can return an error.
    fn linear_solve_matrix(a_mat: &Matrix<f32>, b_vec: &Matrix<f32>) -> Result<Matrix<f32>> {
        if a_mat.get_rows() != a_mat.get_cols() || a_mat.get_cols() != b_vec.get_rows() || b_vec.get_cols() != 1 {
            return Err(Jeeperr::DimensionError)
        }
        let dim_n: usize = a_mat.get_rows();
    
        let lu_matrix: Matrix<f32> = Matrix::lu_decomp(a_mat)?;
    
        let mut y_data: Vec<f32> = vec![0.0; dim_n];
        for idx in 0..dim_n {
            let sum: f32 = (0..idx)
                .into_iter()
                .map(|kdx| lu_matrix[[idx, kdx]] * y_data[kdx])
                .sum();
    
            y_data[idx] = b_vec.get_data()[idx] - sum;
        }
    
        let mut x_data: Vec<f32> = vec![0.0; dim_n];
        for idx in (0..dim_n).rev() {
            let sum: f32 = ((idx + 1)..dim_n)
                .into_iter()
                .map(|kdx| lu_matrix[[idx, kdx]] * x_data[kdx])
                .sum();
    
            x_data[idx] = (y_data[idx] - sum) / lu_matrix[[idx, idx]];
        }
    
        return Ok(Matrix::new(x_data, dim_n, 1)?);
    }

    /// Find absolute maximum value of matrix.
    /// This will find and output the single maximum value across all elements in the matrix.
    fn max(matrix: &Matrix<f32>) -> f32 {
        let max_val: f32 = matrix.get_data()
            .iter()
            .fold(f32::MIN, |left, &right| left.max(right));
    
        return max_val
    }

    /// Find maximum value of matrix along an axis.
    /// This will find the maximum value of each row or column and output them as a matrix.
    fn axis_max(matrix: &Matrix<f32>, axis: Axis) -> Matrix<f32> {
        match axis {
            Axis::Row => {
                let mut max_data: Vec<f32> = Vec::with_capacity(matrix.get_rows());
    
                for row in matrix.all_rows() {
                    max_data.push(Matrix::max(&row));
                }
    
                return Matrix::new(max_data, matrix.get_rows(), 1).unwrap()
            },
            Axis::Col => {
                let mut max_data: Vec<f32> = Vec::with_capacity(matrix.get_cols());
    
                for col in matrix.all_cols() {
                    max_data.push(Matrix::max(&col));
                }
    
                return Matrix::new(max_data, 1, matrix.get_cols()).unwrap()
            }
        }
    }

    /// Find absolute minimum value of matrix.
    /// This will find and output the single minimum value across all elements in the matrix.
    fn min(matrix: &Matrix<f32>) -> f32 {
        let min_val: f32 = matrix.get_data()
            .iter()
            .fold(f32::MAX, |left, &right| left.min(right));
    
        return min_val
    }

    /// Find minimum value of matrix along an axis.
    /// This will find the minimum value of each row or column and output them as a matrix.
    fn axis_min(matrix: &Matrix<f32>, axis: Axis) -> Matrix<f32> {
        match axis {
            Axis::Row => {
                let mut min_data: Vec<f32> = Vec::with_capacity(matrix.get_rows());
    
                for row in matrix.all_rows() {
                    min_data.push(Matrix::min(&row));
                }
    
                return Matrix::new(min_data, matrix.get_rows(), 1).unwrap()
            },
            Axis::Col => {
                let mut min_data: Vec<f32> = Vec::with_capacity(matrix.get_cols());
    
                for col in matrix.all_cols() {
                    min_data.push(Matrix::min(&col));
                }
    
                return Matrix::new(min_data, 1, matrix.get_cols()).unwrap()
            }
        }
    }

    /// Find sum of all elements in a matrix.
    fn sum(matrix: &Matrix<f32>) -> f32 {
        let sum: f32 = matrix.get_data()
            .iter()
            .sum();
    
        return sum
    }

    /// Sum elements of a matrix along rows or columns, returning a column or row matrix.
    fn axis_sum(matrix: &Matrix<f32>, axis: Axis) -> Matrix<f32> {
        match axis {
            Axis::Row => {
                let mut sum_data: Vec<f32> = Vec::with_capacity(matrix.get_rows());

                for row in matrix.all_rows() {
                    sum_data.push(Matrix::sum(&row));
                }

                return Matrix::new(sum_data, matrix.get_rows(), 1).unwrap()
            },
            Axis::Col => {
                let mut sum_data: Vec<f32> = Vec::with_capacity(matrix.get_cols());

                for col in matrix.all_cols() {
                    sum_data.push(Matrix::sum(&col));
                }

                return Matrix::new(sum_data, 1, matrix.get_cols()).unwrap()
            }
        }
    }

    /// Concatenate slice of matrices together along an axis.
    /// Think of this as "stacking" the matrices together either top to bottom (Axis::Row) or left to right (Axis::Col)
    fn concatenate(matrices: &[Matrix<f32>], axis: Axis) -> Result<Matrix<f32>> {
        match axis {
            Axis::Row => {
                if matrices.iter().any(|mat| mat.get_cols() != matrices[0].get_cols()) {
                    return Err(Jeeperr::DimensionError)
                }
    
                let output_rows: usize = matrices.iter().map(|mat| mat.get_rows()).sum();
                let output_cols: usize = matrices[0].get_cols();
                let mut output_data: Vec<f32> = matrices[0].get_data().to_vec();
                for matrix_idx in 1..matrices.len() {
                    output_data.extend(matrices[matrix_idx].get_data().iter())
                }
    
                return Ok(Matrix::new(output_data, output_rows, output_cols)?)
            },
            Axis::Col => {
                if matrices.iter().any(|mat| mat.get_rows() != matrices[0].get_rows()) {
                    return Err(Jeeperr::DimensionError)
                }
    
                let output_rows: usize = matrices[0].get_rows();
                let output_cols: usize = matrices.iter().map(|mat| mat.get_cols()).sum();
                let mut output_data: Vec<f32> = Vec::with_capacity(output_rows * output_cols);
    
                for output_row in 0..output_rows {
                    for matrix in matrices {
                        output_data.extend((*matrix.row_vec(output_row)?).iter());
                    }
                }
    
                return Ok(Matrix::new(output_data, output_rows, output_cols)?)
            }
        }
    }
}

impl MatrixUtilities<f64> for Matrix<f64> {
    /// Perform partial LU decomposition of a matrix.
    /// This function returns a single matrix LU that is equivalent to L + U - I.
    /// This is designed primarily as a helper function for linear_solve_matrix.
    fn lu_decomp(matrix: &Matrix<f64>) -> Result<Matrix<f64>> {
        if matrix.get_rows() != matrix.get_cols() {
            return Err(Jeeperr::DimensionError)
        }
        let dim_n: usize = matrix.get_rows();
    
        let mut lu_data: Vec<f64> = vec![0.0; dim_n * dim_n];
        for idx in 0..dim_n {
            for jdx in idx..dim_n {
                let sum: f64 = (0..idx)
                    .into_iter()
                    .map(|kdx| lu_data[idx * dim_n + kdx] * lu_data[kdx * dim_n + jdx])
                    .sum();
                lu_data[idx * dim_n + jdx] = matrix[[idx, jdx]] - sum;
            }
            for jdx in (idx + 1)..dim_n {
                let sum: f64 = (0..idx)
                    .into_iter()
                    .map(|kdx| lu_data[jdx * dim_n + kdx] * lu_data[kdx * dim_n + idx])
                    .sum();
                lu_data[jdx * dim_n + idx] = (matrix[[jdx, idx]] - sum) / lu_data[idx * dim_n + idx];
            }
        }
    
        return Ok(Matrix::new(lu_data, dim_n, dim_n)?)
    }

    /// Solve system of linear equations Ax=b.
    /// This will output x as a Matrix given A and b.
    /// This function can return an error.
    fn linear_solve_matrix(a_mat: &Matrix<f64>, b_vec: &Matrix<f64>) -> Result<Matrix<f64>> {
        if a_mat.get_rows() != a_mat.get_cols() || a_mat.get_cols() != b_vec.get_rows() || b_vec.get_cols() != 1 {
            return Err(Jeeperr::DimensionError)
        }
        let dim_n: usize = a_mat.get_rows();
    
        let lu_matrix: Matrix<f64> = Matrix::lu_decomp(a_mat)?;
    
        let mut y_data: Vec<f64> = vec![0.0; dim_n];
        for idx in 0..dim_n {
            let sum: f64 = (0..idx)
                .into_iter()
                .map(|kdx| lu_matrix[[idx, kdx]] * y_data[kdx])
                .sum();
    
            y_data[idx] = b_vec.get_data()[idx] - sum;
        }
    
        let mut x_data: Vec<f64> = vec![0.0; dim_n];
        for idx in (0..dim_n).rev() {
            let sum: f64 = ((idx + 1)..dim_n)
                .into_iter()
                .map(|kdx| lu_matrix[[idx, kdx]] * x_data[kdx])
                .sum();
    
            x_data[idx] = (y_data[idx] - sum) / lu_matrix[[idx, idx]];
        }
    
        return Ok(Matrix::new(x_data, dim_n, 1)?);
    }

    /// Find absolute maximum value of matrix.
    /// This will find and output the single maximum value across all elements in the matrix.
    fn max(matrix: &Matrix<f64>) -> f64 {
        let max_val: f64 = matrix.get_data()
            .iter()
            .fold(f64::MIN, |left, &right| left.max(right));
    
        return max_val
    }

    /// Find maximum value of matrix along an axis.
    /// This will find the maximum value of each row or column and output them as a matrix.
    fn axis_max(matrix: &Matrix<f64>, axis: Axis) -> Matrix<f64> {
        match axis {
            Axis::Row => {
                let mut max_data: Vec<f64> = Vec::with_capacity(matrix.get_rows());
    
                for row in matrix.all_rows() {
                    max_data.push(Matrix::max(&row));
                }
    
                return Matrix::new(max_data, matrix.get_rows(), 1).unwrap()
            },
            Axis::Col => {
                let mut max_data: Vec<f64> = Vec::with_capacity(matrix.get_cols());
    
                for col in matrix.all_cols() {
                    max_data.push(Matrix::max(&col));
                }
    
                return Matrix::new(max_data, 1, matrix.get_cols()).unwrap()
            }
        }
    }

    /// Find absolute minimum value of matrix.
    /// This will find the minimum value of each row or column and output them as a matrix.
    fn min(matrix: &Matrix<f64>) -> f64 {
        let min_val: f64 = matrix.get_data()
            .iter()
            .fold(f64::MAX, |left, &right| left.min(right));
    
        return min_val
    }

    /// Find minimum value of matrix along an axis.
    fn axis_min(matrix: &Matrix<f64>, axis: Axis) -> Matrix<f64> {
        match axis {
            Axis::Row => {
                let mut min_data: Vec<f64> = Vec::with_capacity(matrix.get_rows());
    
                for row in matrix.all_rows() {
                    min_data.push(Matrix::min(&row));
                }
    
                return Matrix::new(min_data, matrix.get_rows(), 1).unwrap()
            },
            Axis::Col => {
                let mut min_data: Vec<f64> = Vec::with_capacity(matrix.get_cols());
    
                for col in matrix.all_cols() {
                    min_data.push(Matrix::min(&col));
                }
    
                return Matrix::new(min_data, 1, matrix.get_cols()).unwrap()
            }
        }
    }

    /// Find sum of all elements in a matrix.
    fn sum(matrix: &Matrix<f64>) -> f64 {
        let sum: f64 = matrix.get_data()
            .iter()
            .sum();
    
        return sum
    }

    /// Sum elements of a matrix along rows or columns, returning a column or row matrix.
    fn axis_sum(matrix: &Matrix<f64>, axis: Axis) -> Matrix<f64> {
        match axis {
            Axis::Row => {
                let mut sum_data: Vec<f64> = Vec::with_capacity(matrix.get_rows());

                for row in matrix.all_rows() {
                    sum_data.push(Matrix::sum(&row));
                }

                return Matrix::new(sum_data, matrix.get_rows(), 1).unwrap()
            },
            Axis::Col => {
                let mut sum_data: Vec<f64> = Vec::with_capacity(matrix.get_cols());

                for col in matrix.all_cols() {
                    sum_data.push(Matrix::sum(&col));
                }

                return Matrix::new(sum_data, 1, matrix.get_cols()).unwrap()
            }
        }
    }

    /// Concatenate slice of matrices together along an axis.
    /// Think of this as "stacking" the matrices together either top to bottom (Axis::Row) or left to right (Axis::Col)
    fn concatenate(matrices: &[Matrix<f64>], axis: Axis) -> Result<Matrix<f64>> {
        match axis {
            Axis::Row => {
                if matrices.iter().any(|mat| mat.get_cols() != matrices[0].get_cols()) {
                    return Err(Jeeperr::DimensionError)
                }
    
                let output_rows: usize = matrices.iter().map(|mat| mat.get_rows()).sum();
                let output_cols: usize = matrices[0].get_cols();
                let mut output_data: Vec<f64> = matrices[0].get_data().to_vec();
                for matrix_idx in 1..matrices.len() {
                    output_data.extend(matrices[matrix_idx].get_data().iter())
                }
    
                return Ok(Matrix::new(output_data, output_rows, output_cols)?)
            },
            Axis::Col => {
                if matrices.iter().any(|mat| mat.get_rows() != matrices[0].get_rows()) {
                    return Err(Jeeperr::DimensionError)
                }
    
                let output_rows: usize = matrices[0].get_rows();
                let output_cols: usize = matrices.iter().map(|mat| mat.get_cols()).sum();
                let mut output_data: Vec<f64> = Vec::with_capacity(output_rows * output_cols);
    
                for output_row in 0..output_rows {
                    for matrix in matrices {
                        output_data.extend((*matrix.row_vec(output_row)?).iter());
                    }
                }
    
                return Ok(Matrix::new(output_data, output_rows, output_cols)?)
            }
        }
    }
}