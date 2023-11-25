use crate::IsFloat;
use crate::Result;
use crate::Jeeperr;
use crate::linalg::Matrix;

/// Trait containing boolean operations for the Matrix struct.
pub trait BooleanMatrixOperations<T: IsFloat + std::fmt::Debug + Copy + Clone> {
    /// Return boolean vector with one element for each matrix element
    /// with values based on element == number
    fn equal(matrix: &Matrix<T>, number: T) -> Vec<bool>;

    /// Return boolean vector with one element for each matrix element
    /// with values based on element < number
    fn less_than(matrix: &Matrix<T>, number: T) -> Vec<bool>;
    /// Return boolean vector with one element for each matrix element
    /// with values based on element <= number
    fn less_equal(matrix: &Matrix<T>, number: T) -> Vec<bool>;

    /// Return boolean vector with one element for each matrix element
    /// with values based on element > number
    fn greater_than(matrix: &Matrix<T>, number: T) -> Vec<bool>;
    /// Return boolean vector with one element for each matrix element
    /// with values based on element >= number
    fn greater_equal(matrix: &Matrix<T>, number: T) -> Vec<bool>;

    /// Return boolean vector with one element for each matrix element
    /// with values based on left_element == right_element
    /// 
    /// Returns error if left and right matrices have different dimensions
    fn equal_mat(left_matrix: &Matrix<T>, right_matrix: &Matrix<T>) -> Result<Vec<bool>>;

    /// Return boolean vector with one element for each matrix element
    /// with values based on left_element < right_element
    /// 
    /// Returns error if left and right matrices have different dimensions
    fn less_than_mat(left_matrix: &Matrix<T>, right_matrix: &Matrix<T>) -> Result<Vec<bool>>;
    /// Return boolean vector with one element for each matrix element
    /// with values based on left_element <= right_element
    /// 
    /// Returns error if left and right matrices have different dimensions
    fn less_equal_mat(left_matrix: &Matrix<T>, right_matrix: &Matrix<T>) -> Result<Vec<bool>>;

    /// Return boolean vector with one element for each matrix element
    /// with values based on left_element > right_element
    /// 
    /// Returns error if left and right matrices have different dimensions
    fn greater_than_mat(left_matrix: &Matrix<T>, right_matrix: &Matrix<T>) -> Result<Vec<bool>>;
    /// Return boolean vector with one element for each matrix element
    /// with values based on left_element >= right_element
    /// 
    /// Returns error if left and right matrices have different dimensions
    fn greater_equal_mat(left_matrix: &Matrix<T>, right_matrix: &Matrix<T>) -> Result<Vec<bool>>;
}

impl BooleanMatrixOperations<f32> for Matrix<f32> {
    /// Return boolean vector with one element for each matrix element
    /// with values based on element == number
    fn equal(matrix: &Matrix<f32>, number: f32) -> Vec<bool> {
        let output_bools: Vec<bool> = matrix.get_data().into_iter()
            .map(|&item| item == number)
            .collect::<Vec<bool>>();

        return output_bools
    }

    /// Return boolean vector with one element for each matrix element
    /// with values based on element < number
    fn less_than(matrix: &Matrix<f32>, number: f32) -> Vec<bool> {
        let output_bools: Vec<bool> = matrix.get_data().into_iter()
            .map(|&item| item < number)
            .collect::<Vec<bool>>();

        return output_bools
    }

    /// Return boolean vector with one element for each matrix element
    /// with values based on element <= number
    fn less_equal(matrix: &Matrix<f32>, number: f32) -> Vec<bool> {
        let output_bools: Vec<bool> = matrix.get_data().into_iter()
            .map(|&item| item <= number)
            .collect::<Vec<bool>>();

        return output_bools
    }

    /// Return boolean vector with one element for each matrix element
    /// with values based on element > number
    fn greater_than(matrix: &Matrix<f32>, number: f32) -> Vec<bool> {
        let output_bools: Vec<bool> = matrix.get_data().into_iter()
            .map(|&item| item > number)
            .collect::<Vec<bool>>();

        return output_bools
    }

    /// Return boolean vector with one element for each matrix element
    /// with values based on element >= number
    fn greater_equal(matrix: &Matrix<f32>, number: f32) -> Vec<bool> {
        let output_bools: Vec<bool> = matrix.get_data().into_iter()
            .map(|&item| item >= number)
            .collect::<Vec<bool>>();

        return output_bools
    }

    /// Return boolean vector with one element for each matrix element
    /// with values based on left_element == right_element
    /// 
    /// Returns error if left and right matrices have different dimensions
    fn equal_mat(left_matrix: &Matrix<f32>, right_matrix: &Matrix<f32>) -> Result<Vec<bool>> {
        if left_matrix.get_rows() != right_matrix.get_rows() || left_matrix.get_cols() != right_matrix.get_cols() {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = left_matrix.get_rows() * left_matrix.get_cols();

        let output_bools: Vec<bool> = (0..n_elements).into_iter()
            .map(|idx| left_matrix.lindex(idx) == right_matrix.lindex(idx))
            .collect::<Vec<bool>>();

        return Ok(output_bools)
    }

    /// Return boolean vector with one element for each matrix element
    /// with values based on left_element < right_element
    /// 
    /// Returns error if left and right matrices have different dimensions
    fn less_than_mat(left_matrix: &Matrix<f32>, right_matrix: &Matrix<f32>) -> Result<Vec<bool>> {
        if left_matrix.get_rows() != right_matrix.get_rows() || left_matrix.get_cols() != right_matrix.get_cols() {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = left_matrix.get_rows() * left_matrix.get_cols();

        let output_bools: Vec<bool> = (0..n_elements).into_iter()
            .map(|idx| left_matrix.lindex(idx) < right_matrix.lindex(idx))
            .collect::<Vec<bool>>();

        return Ok(output_bools)
    }

    /// Return boolean vector with one element for each matrix element
    /// with values based on left_element <= right_element
    /// 
    /// Returns error if left and right matrices have different dimensions
    fn less_equal_mat(left_matrix: &Matrix<f32>, right_matrix: &Matrix<f32>) -> Result<Vec<bool>> {
        if left_matrix.get_rows() != right_matrix.get_rows() || left_matrix.get_cols() != right_matrix.get_cols() {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = left_matrix.get_rows() * left_matrix.get_cols();

        let output_bools: Vec<bool> = (0..n_elements).into_iter()
            .map(|idx| left_matrix.lindex(idx) <= right_matrix.lindex(idx))
            .collect::<Vec<bool>>();

        return Ok(output_bools)
    }

    /// Return boolean vector with one element for each matrix element
    /// with values based on left_element > right_element
    /// 
    /// Returns error if left and right matrices have different dimensions
    fn greater_than_mat(left_matrix: &Matrix<f32>, right_matrix: &Matrix<f32>) -> Result<Vec<bool>> {
        if left_matrix.get_rows() != right_matrix.get_rows() || left_matrix.get_cols() != right_matrix.get_cols() {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = left_matrix.get_rows() * left_matrix.get_cols();

        let output_bools: Vec<bool> = (0..n_elements).into_iter()
            .map(|idx| left_matrix.lindex(idx) > right_matrix.lindex(idx))
            .collect::<Vec<bool>>();

        return Ok(output_bools)
    }

    /// Return boolean vector with one element for each matrix element
    /// with values based on left_element >= right_element
    /// 
    /// Returns error if left and right matrices have different dimensions
    fn greater_equal_mat(left_matrix: &Matrix<f32>, right_matrix: &Matrix<f32>) -> Result<Vec<bool>> {
        if left_matrix.get_rows() != right_matrix.get_rows() || left_matrix.get_cols() != right_matrix.get_cols() {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = left_matrix.get_rows() * left_matrix.get_cols();

        let output_bools: Vec<bool> = (0..n_elements).into_iter()
            .map(|idx| left_matrix.lindex(idx) >= right_matrix.lindex(idx))
            .collect::<Vec<bool>>();

        return Ok(output_bools)
    }
}

impl BooleanMatrixOperations<f64> for Matrix<f64> {
    /// Return boolean vector with one element for each matrix element
    /// with values based on element == number
    fn equal(matrix: &Matrix<f64>, number: f64) -> Vec<bool> {
        let output_bools: Vec<bool> = matrix.get_data().into_iter()
            .map(|&item| item == number)
            .collect::<Vec<bool>>();

        return output_bools
    }

    /// Return boolean vector with one element for each matrix element
    /// with values based on element < number
    fn less_than(matrix: &Matrix<f64>, number: f64) -> Vec<bool> {
        let output_bools: Vec<bool> = matrix.get_data().into_iter()
            .map(|&item| item < number)
            .collect::<Vec<bool>>();

        return output_bools
    }

    /// Return boolean vector with one element for each matrix element
    /// with values based on element <= number
    fn less_equal(matrix: &Matrix<f64>, number: f64) -> Vec<bool> {
        let output_bools: Vec<bool> = matrix.get_data().into_iter()
            .map(|&item| item <= number)
            .collect::<Vec<bool>>();

        return output_bools
    }

    /// Return boolean vector with one element for each matrix element
    /// with values based on element > number
    fn greater_than(matrix: &Matrix<f64>, number: f64) -> Vec<bool> {
        let output_bools: Vec<bool> = matrix.get_data().into_iter()
            .map(|&item| item > number)
            .collect::<Vec<bool>>();

        return output_bools
    }

    /// Return boolean vector with one element for each matrix element
    /// with values based on element >= number
    fn greater_equal(matrix: &Matrix<f64>, number: f64) -> Vec<bool> {
        let output_bools: Vec<bool> = matrix.get_data().into_iter()
            .map(|&item| item >= number)
            .collect::<Vec<bool>>();

        return output_bools
    }

    /// Return boolean vector with one element for each matrix element
    /// with values based on left_element == right_element
    /// 
    /// Returns error if left and right matrices have different dimensions
    fn equal_mat(left_matrix: &Matrix<f64>, right_matrix: &Matrix<f64>) -> Result<Vec<bool>> {
        if left_matrix.get_rows() != right_matrix.get_rows() || left_matrix.get_cols() != right_matrix.get_cols() {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = left_matrix.get_rows() * left_matrix.get_cols();

        let output_bools: Vec<bool> = (0..n_elements).into_iter()
            .map(|idx| left_matrix.lindex(idx) == right_matrix.lindex(idx))
            .collect::<Vec<bool>>();

        return Ok(output_bools)
    }

    /// Return boolean vector with one element for each matrix element
    /// with values based on left_element < right_element
    /// 
    /// Returns error if left and right matrices have different dimensions
    fn less_than_mat(left_matrix: &Matrix<f64>, right_matrix: &Matrix<f64>) -> Result<Vec<bool>> {
        if left_matrix.get_rows() != right_matrix.get_rows() || left_matrix.get_cols() != right_matrix.get_cols() {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = left_matrix.get_rows() * left_matrix.get_cols();

        let output_bools: Vec<bool> = (0..n_elements).into_iter()
            .map(|idx| left_matrix.lindex(idx) < right_matrix.lindex(idx))
            .collect::<Vec<bool>>();

        return Ok(output_bools)
    }

    /// Return boolean vector with one element for each matrix element
    /// with values based on left_element <= right_element
    /// 
    /// Returns error if left and right matrices have different dimensions
    fn less_equal_mat(left_matrix: &Matrix<f64>, right_matrix: &Matrix<f64>) -> Result<Vec<bool>> {
        if left_matrix.get_rows() != right_matrix.get_rows() || left_matrix.get_cols() != right_matrix.get_cols() {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = left_matrix.get_rows() * left_matrix.get_cols();

        let output_bools: Vec<bool> = (0..n_elements).into_iter()
            .map(|idx| left_matrix.lindex(idx) <= right_matrix.lindex(idx))
            .collect::<Vec<bool>>();

        return Ok(output_bools)
    }

    /// Return boolean vector with one element for each matrix element
    /// with values based on left_element > right_element
    /// 
    /// Returns error if left and right matrices have different dimensions
    fn greater_than_mat(left_matrix: &Matrix<f64>, right_matrix: &Matrix<f64>) -> Result<Vec<bool>> {
        if left_matrix.get_rows() != right_matrix.get_rows() || left_matrix.get_cols() != right_matrix.get_cols() {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = left_matrix.get_rows() * left_matrix.get_cols();

        let output_bools: Vec<bool> = (0..n_elements).into_iter()
            .map(|idx| left_matrix.lindex(idx) > right_matrix.lindex(idx))
            .collect::<Vec<bool>>();

        return Ok(output_bools)
    }

    /// Return boolean vector with one element for each matrix element
    /// with values based on left_element >= right_element
    /// 
    /// Returns error if left and right matrices have different dimensions
    fn greater_equal_mat(left_matrix: &Matrix<f64>, right_matrix: &Matrix<f64>) -> Result<Vec<bool>> {
        if left_matrix.get_rows() != right_matrix.get_rows() || left_matrix.get_cols() != right_matrix.get_cols() {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = left_matrix.get_rows() * left_matrix.get_cols();

        let output_bools: Vec<bool> = (0..n_elements).into_iter()
            .map(|idx| left_matrix.lindex(idx) >= right_matrix.lindex(idx))
            .collect::<Vec<bool>>();

        return Ok(output_bools)
    }
}