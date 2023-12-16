use crate::IsFloat;
use crate::Result;
use crate::Jeeperr;
use crate::linalg::Matrix;

/// Trait containing reference based arithmatic operations for the Matrix struct.
pub trait ReferenceOperations<T: IsFloat + std::fmt::Debug + Copy + Clone> {
    /// Add one matrix to another matrix.
    /// Inputs are immutable references unlike the basic addition operator.
    fn ref_add(left: &Matrix<T>, right: &Matrix<T>) -> Result<Matrix<T>>;
    /// Negate a matrix.
    /// Input is an immutable reference unlike the basic negation operator.
    fn ref_neg(mat: &Matrix<T>) -> Matrix<T>;
    /// Subtract one matrix from another matrix.
    /// Inputs are immutable references unlike the basic subtraction operator.
    fn ref_sub(left: &Matrix<T>, right: &Matrix<T>) -> Result<Matrix<T>>;
    /// Multiply one matrix by another matrix.
    /// Inputs are immutable references unlike the basic multiplication operator.
    fn ref_mul(left: &Matrix<T>, right: &Matrix<T>) -> Result<Matrix<T>>;
    fn ref_flmul(left: &T, right: &Matrix<T>) -> Matrix<T>;
    /// Multiply two matrices elementwise. 
    /// Inputs are immutable references unlike the basic elementwise function.
    fn ref_ewm(left: &Matrix<T>, right: &Matrix<T>) -> Result<Matrix<T>>;
    /// Divide two matrices elementwise.
    /// Inputs are immutable references unlike the basic elementwise_divide function.
    fn ref_ewd(left: &Matrix<T>, right: &Matrix<T>) -> Result<Matrix<T>>;
    fn ref_dot(left: &Matrix<T>, right: &Matrix<T>) -> Result<T>;
}

impl ReferenceOperations<f32> for Matrix<f32> {
    /// Add one matrix to another matrix.
    /// Inputs are immutable references unlike the basic addition operator.
    fn ref_add(left: &Matrix<f32>, right: &Matrix<f32>) -> Result<Matrix<f32>> {
        if (left.get_rows() != right.get_rows()) || (left.get_cols() != right.get_cols()) {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = left.get_rows() * left.get_cols();
        let output_data: Vec<f32> = (0..n_elements).into_iter()
            .map(|idx| left.lindex(idx) + right.lindex(idx))
            .collect::<Vec<f32>>();

        let output_mat: Matrix<f32> = Matrix::new(output_data, left.get_rows(), left.get_cols()).unwrap();
        Ok(output_mat)
    }

    /// Negate a matrix.
    /// Input is an immutable reference unlike the basic negation operator.
    fn ref_neg(mat: &Matrix<f32>) -> Matrix<f32> {
        let n_elements: usize = mat.get_rows() * mat.get_cols();
        let output_data: Vec<f32> = (0..n_elements).into_iter()
            .map(|idx| -mat.lindex(idx))
            .collect::<Vec<f32>>();

        let output_mat: Matrix<f32> = Matrix::new(output_data, mat.get_rows(), mat.get_cols()).unwrap();
        return output_mat
    }

    /// Subtract one matrix from another matrix.
    /// Inputs are immutable references unlike the basic subtraction operator.
    fn ref_sub(left: &Matrix<f32>, right: &Matrix<f32>) -> Result<Matrix<f32>> {
        if (left.get_rows() != right.get_rows()) || (left.get_cols() != right.get_cols()) {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = left.get_rows() * left.get_cols();
        let output_data: Vec<f32> = (0..n_elements).into_iter()
            .map(|idx| left.lindex(idx) - right.lindex(idx))
            .collect::<Vec<f32>>();

        let output_mat: Matrix<f32> = Matrix::new(output_data, left.get_rows(), left.get_cols()).unwrap();
        Ok(output_mat)
    }

    /// Multiply one matrix by another matrix.
    /// Inputs are immutable references unlike the basic multiplication operator.
    fn ref_mul(left: &Matrix<f32>, right: &Matrix<f32>) -> Result<Matrix<f32>> {
        if left.get_cols() != right.get_rows() {
            return Err(Jeeperr::DimensionError)
        }

        let new_n_elements: usize = left.get_rows() * right.get_cols();
        let mut output_data: Vec<f32> = Vec::with_capacity(new_n_elements);

        for lhs_row in 0..left.get_rows() {
            for rhs_col in 0..right.get_cols() {
                let dot_prod: f32 = (0..left.get_cols()).into_iter()
                    .map(|idx| left[[lhs_row, idx]] * right[[idx, rhs_col]])
                    .sum();

                output_data.push(dot_prod);
            }
        }

        let output_mat: Matrix<f32> = Matrix::new(output_data, left.get_rows(), right.get_cols())?;
        Ok(output_mat)
    }

    fn ref_flmul(left: &f32, right: &Matrix<f32>) -> Matrix<f32> { return Matrix::<f32>::zeros(1, 1) }

    /// Multiply two matrices elementwise. 
    /// Inputs are immutable references unlike the basic elementwise function.
    fn ref_ewm(left: &Matrix<f32>, right: &Matrix<f32>) -> Result<Matrix<f32>> {
        if (left.get_rows() != right.get_rows()) || (left.get_cols() != right.get_cols()) {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = left.get_rows() * left.get_cols();
        let output_data: Vec<f32> = (0..n_elements).into_iter()
            .map(|idx| left.lindex(idx) * right.lindex(idx))
            .collect::<Vec<f32>>();

        let output_mat: Matrix<f32> = Matrix::new(output_data, left.get_rows(), left.get_cols()).unwrap();
        Ok(output_mat)
    }

    /// Divide two matrices elementwise.
    /// Inputs are immutable references unlike the basic elementwise_divide function.
    fn ref_ewd(left: &Matrix<f32>, right: &Matrix<f32>) -> Result<Matrix<f32>> {
        if (left.get_rows() != right.get_rows()) || (left.get_cols() != right.get_cols()) {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = left.get_rows() * left.get_cols();
        let output_data: Vec<f32> = (0..n_elements).into_iter()
            .map(|idx| left.lindex(idx) / right.lindex(idx))
            .collect::<Vec<f32>>();

        let output_mat: Matrix<f32> = Matrix::new(output_data, left.get_rows(), left.get_cols()).unwrap();
        Ok(output_mat)
    }

    fn ref_dot(left: &Matrix<f32>, right: &Matrix<f32>) -> Result<f32> {
        if (left.get_rows() != right.get_rows()) || (left.get_cols() != right.get_cols()) {
            return Err(Jeeperr::DimensionError)
        }

        if left.get_rows() != 1 || left.get_cols() != 1 {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = left.get_rows() * left.get_cols();

        return Ok((0..n_elements).into_iter()
            .map(|idx| left.lindex(idx) * right.lindex(idx))
            .sum())
    }
}

impl ReferenceOperations<f64> for Matrix<f64> {
    /// Add one matrix to another matrix.
    /// Inputs are immutable references unlike the basic addition operator.
    fn ref_add(left: &Matrix<f64>, right: &Matrix<f64>) -> Result<Matrix<f64>> {
        if (left.get_rows() != right.get_rows()) || (left.get_cols() != right.get_cols()) {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = left.get_rows() * left.get_cols();
        let output_data: Vec<f64> = (0..n_elements).into_iter()
            .map(|idx| left.lindex(idx) + right.lindex(idx))
            .collect::<Vec<f64>>();

        let output_mat: Matrix<f64> = Matrix::new(output_data, left.get_rows(), left.get_cols()).unwrap();
        Ok(output_mat)
    }

    /// Negate a matrix.
    /// Input is an immutable reference unlike the basic negation operator.
    fn ref_neg(mat: &Matrix<f64>) -> Matrix<f64> {
        let n_elements: usize = mat.get_rows() * mat.get_cols();
        let output_data: Vec<f64> = (0..n_elements).into_iter()
            .map(|idx| -mat.lindex(idx))
            .collect::<Vec<f64>>();

        let output_mat: Matrix<f64> = Matrix::new(output_data, mat.get_rows(), mat.get_cols()).unwrap();
        return output_mat
    }

    /// Subtract one matrix from another matrix.
    /// Inputs are immutable references unlike the basic subtraction operator.
    fn ref_sub(left: &Matrix<f64>, right: &Matrix<f64>) -> Result<Matrix<f64>> {
        if (left.get_rows() != right.get_rows()) || (left.get_cols() != right.get_cols()) {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = left.get_rows() * left.get_cols();
        let output_data: Vec<f64> = (0..n_elements).into_iter()
            .map(|idx| left.lindex(idx) - right.lindex(idx))
            .collect::<Vec<f64>>();

        let output_mat: Matrix<f64> = Matrix::new(output_data, left.get_rows(), left.get_cols()).unwrap();
        Ok(output_mat)
    }

    /// Multiply one matrix by another matrix.
    /// Inputs are immutable references unlike the basic multiplication operator.
    fn ref_mul(left: &Matrix<f64>, right: &Matrix<f64>) -> Result<Matrix<f64>> {
        if left.get_cols() != right.get_rows() {
            return Err(Jeeperr::DimensionError)
        }

        let new_n_elements: usize = left.get_rows() * right.get_cols();
        let mut output_data: Vec<f64> = Vec::with_capacity(new_n_elements);

        for lhs_row in 0..left.get_rows() {
            for rhs_col in 0..right.get_cols() {
                let dot_prod: f64 = (0..left.get_cols()).into_iter()
                    .map(|idx| left[[lhs_row, idx]] * right[[idx, rhs_col]])
                    .sum();

                output_data.push(dot_prod);
            }
        }

        let output_mat: Matrix<f64> = Matrix::new(output_data, left.get_rows(), right.get_cols())?;
        Ok(output_mat)
    }

    fn ref_flmul(left: &f64, right: &Matrix<f64>) -> Matrix<f64> {
        let output_data: Vec<f64> = right.get_data().iter()
            .map(|val| left * val)
            .collect::<Vec<f64>>();

        return Matrix::new(output_data, right.get_rows(), right.get_cols()).unwrap();
    }

    /// Multiply two matrices elementwise. 
    /// Inputs are immutable references unlike the basic elementwise function.
    fn ref_ewm(left: &Matrix<f64>, right: &Matrix<f64>) -> Result<Matrix<f64>> {
        if (left.get_rows() != right.get_rows()) || (left.get_cols() != right.get_cols()) {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = left.get_rows() * left.get_cols();
        let output_data: Vec<f64> = (0..n_elements).into_iter()
            .map(|idx| left.lindex(idx) * right.lindex(idx))
            .collect::<Vec<f64>>();

        let output_mat: Matrix<f64> = Matrix::new(output_data, left.get_rows(), left.get_cols()).unwrap();
        Ok(output_mat)
    }

    /// Divide two matrices elementwise.
    /// Inputs are immutable references unlike the basic elementwise_divide function.
    fn ref_ewd(left: &Matrix<f64>, right: &Matrix<f64>) -> Result<Matrix<f64>> {
        if (left.get_rows() != right.get_rows()) || (left.get_cols() != right.get_cols()) {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = left.get_rows() * left.get_cols();
        let output_data: Vec<f64> = (0..n_elements).into_iter()
            .map(|idx| left.lindex(idx) / right.lindex(idx))
            .collect::<Vec<f64>>();

        let output_mat: Matrix<f64> = Matrix::new(output_data, left.get_rows(), left.get_cols()).unwrap();
        Ok(output_mat)
    }

    fn ref_dot(left: &Matrix<f64>, right: &Matrix<f64>) -> Result<f64> {
        if (left.get_rows() != right.get_rows()) || (left.get_cols() != right.get_cols()) {
            return Err(Jeeperr::DimensionError)
        }

        if left.get_rows() != 1 && left.get_cols() != 1 {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = left.get_rows() * left.get_cols();

        return Ok((0..n_elements).into_iter()
            .map(|idx| left.lindex(idx) * right.lindex(idx))
            .sum())
    }
}