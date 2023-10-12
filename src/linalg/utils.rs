use crate::Result;
use crate::Jeeperr;
use crate::linalg::Matrix;

/// Axis enum for use in utility functions
pub enum Axis {
    Row,
    Col
}

/// Dot product between two float vectors
pub fn dot(a: Vec<f32>, b: Vec<f32>) -> Result<f32> {
    if a.len() != b.len() {
        return Err(Jeeperr::DimensionError)
    }

    let dot_prod: f32 = (0..a.len()).into_iter()
        .map(|idx| a[idx] * b[idx])
        .sum();
    return Ok(dot_prod)
}

pub fn lu_decomp(matrix: &Matrix<f32>) -> Result<Matrix<f32>> {
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

pub fn linear_solve_matrix(a_mat: &Matrix<f32>, b_vec: &Matrix<f32>) -> Result<Matrix<f32>> {
    if a_mat.get_rows() != a_mat.get_cols() || a_mat.get_cols() != b_vec.get_rows() || b_vec.get_cols() != 1 {
        return Err(Jeeperr::DimensionError)
    }
    let dim_n: usize = a_mat.get_rows();

    let lu_matrix: Matrix<f32> = lu_decomp(a_mat)?;

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

/// Max value of a matrix
pub fn max(matrix: &Matrix<f32>) -> f32 {
    let max_val: f32 = matrix.get_data()
        .iter()
        .fold(f32::MIN, |left, &right| left.max(right));

    return max_val
}

/// Max values of a matrix along an axis
pub fn axis_max(matrix: &Matrix<f32>, axis: Axis) -> Matrix<f32> {
    match axis {
        Axis::Row => {
            let mut max_data: Vec<f32> = Vec::with_capacity(matrix.get_rows());

            for row in 0..matrix.get_rows() {
                let start_idx: usize = row * matrix.get_cols();
                let end_idx: usize = (row + 1) * matrix.get_cols();

                max_data.push(
                    matrix.get_data()[start_idx..end_idx]
                        .iter()
                        .fold(f32::MIN, |left, &right| left.max(right))
                );
            }

            return Matrix::new(max_data, matrix.get_rows(), 1).unwrap()
        },
        Axis::Col => {
            let mut max_data: Vec<f32> = Vec::with_capacity(matrix.get_cols());

            for col in 0..matrix.get_cols() {
                max_data.push(
                    matrix.col_vec(col)
                        .unwrap()
                        .iter()
                        .fold(f32::MIN, |left, &right| left.max(right))
                );
            }

            return Matrix::new(max_data, 1, matrix.get_cols()).unwrap()
        }
    }
}

/// Min value of a matrix
pub fn min(matrix: &Matrix<f32>) -> f32 {
    let min_val: f32 = matrix.get_data()
        .iter()
        .fold(f32::MAX, |left, &right| left.min(right));

    return min_val
}

/// Min values of a matrix along an axis
pub fn axis_min(matrix: &Matrix<f32>, axis: Axis) -> Matrix<f32> {
    match axis {
        Axis::Row => {
            let mut min_data: Vec<f32> = Vec::with_capacity(matrix.get_rows());

            for row in 0..matrix.get_rows() {
                let start_idx: usize = row * matrix.get_cols();
                let end_idx: usize = (row + 1) * matrix.get_cols();

                min_data.push(
                    matrix.get_data()[start_idx..end_idx]
                        .iter()
                        .fold(f32::MAX, |left, &right| left.min(right))
                );
            }

            return Matrix::new(min_data, matrix.get_rows(), 1).unwrap()
        },
        Axis::Col => {
            let mut min_data: Vec<f32> = Vec::with_capacity(matrix.get_cols());

            for col in 0..matrix.get_cols() {
                min_data.push(
                    matrix.col_vec(col)
                        .unwrap()
                        .iter()
                        .fold(f32::MAX, |left, &right| left.min(right))
                );
            }

            return Matrix::new(min_data, 1, matrix.get_cols()).unwrap()
        }
    }
}

/// Concatenate a slice of matrices together
pub fn concatenate(matrices: &[Matrix<f32>], axis: Axis) -> Result<Matrix<f32>> {
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