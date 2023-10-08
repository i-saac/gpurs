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

/// Max value of a matrix
pub fn max(matrix: &Matrix) -> f32 {
    let max_val: f32 = matrix.get_data()
        .iter()
        .fold(f32::MIN, |left, &right| left.max(right));

    return max_val
}

/// Max values of a matrix along an axis
pub fn axis_max(matrix: &Matrix, axis: Axis) -> Matrix {
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
pub fn min(matrix: &Matrix) -> f32 {
    let min_val: f32 = matrix.get_data()
        .iter()
        .fold(f32::MAX, |left, &right| left.min(right));

    return min_val
}

/// Min values of a matrix along an axis
pub fn axis_min(matrix: &Matrix, axis: Axis) -> Matrix {
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
pub fn concatenate(matrices: &[Matrix], axis: Axis) -> Result<Matrix> {
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