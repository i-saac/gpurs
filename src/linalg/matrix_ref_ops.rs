use std::ops;

use crate::Result;
use crate::Jeeperr;
use crate::linalg::Matrix;

impl Matrix<f32> {
    pub fn ref_elementwise_multiply(&self, rhs: &Matrix<f32>) -> Result<Matrix<f32>> {
        if self.get_rows() != rhs.get_rows() || self.get_cols() != rhs.get_cols() {
            return Err(Jeeperr::DimensionError)
        }

        let mut ewmult_data: Vec<f32> = Vec::with_capacity(self.get_elements());

        for element in 0..self.get_elements() {
            ewmult_data.push(self.lindex(element) * rhs.lindex(element));
        }

        return Matrix::new(ewmult_data, self.get_rows(), self.get_cols())
    }

    pub fn ref_elementwise_divide(&self, rhs: &Matrix<f32>) -> Result<Matrix<f32>> {
        if self.get_rows() != rhs.get_rows() || self.get_cols() != rhs.get_cols() {
            return Err(Jeeperr::DimensionError)
        }

        let mut ewmult_data: Vec<f32> = Vec::with_capacity(self.get_elements());

        for element in 0..self.get_elements() {
            ewmult_data.push(self.lindex(element) / rhs.lindex(element));
        }

        return Matrix::new(ewmult_data, self.get_rows(), self.get_cols())
    }

    pub fn ref_sumproduct(&self, rhs: &Matrix<f32>) -> Result<f32> {
        if self.get_elements() != rhs.get_elements() {
            return Err(Jeeperr::DimensionError)
        }

        let mut sumprod: f32 = 0.0;
        for element in 0..self.get_elements() {
            sumprod += self.lindex(element) * rhs.lindex(element);
        }

        return Ok(sumprod)
    }
}

impl Matrix<f64> {
    pub fn ref_elementwise_multiply(&self, rhs: &Matrix<f64>) -> Result<Matrix<f64>> {
        if self.get_rows() != rhs.get_rows() || self.get_cols() != rhs.get_cols() {
            return Err(Jeeperr::DimensionError)
        }

        let mut ewmult_data: Vec<f64> = Vec::with_capacity(self.get_elements());

        for element in 0..self.get_elements() {
            ewmult_data.push(self.lindex(element) * rhs.lindex(element));
        }

        return Matrix::new(ewmult_data, self.get_rows(), self.get_cols())
    }

    pub fn ref_elementwise_divide(&self, rhs: &Matrix<f64>) -> Result<Matrix<f64>> {
        if self.get_rows() != rhs.get_rows() || self.get_cols() != rhs.get_cols() {
            return Err(Jeeperr::DimensionError)
        }

        let mut ewmult_data: Vec<f64> = Vec::with_capacity(self.get_elements());

        for element in 0..self.get_elements() {
            ewmult_data.push(self.lindex(element) / rhs.lindex(element));
        }

        return Matrix::new(ewmult_data, self.get_rows(), self.get_cols())
    }

    pub fn ref_sumproduct(&self, rhs: &Matrix<f64>) -> Result<f64> {
        if self.get_elements() != rhs.get_elements() {
            return Err(Jeeperr::DimensionError)
        }

        let mut sumprod: f64 = 0.0;
        for element in 0..self.get_elements() {
            sumprod += self.lindex(element) * rhs.lindex(element);
        }

        return Ok(sumprod)
    }
}

/// Add float to matrix.
/// 
/// ```
/// # use gpurs::linalg::Matrix;
/// let matrix: Matrix<f32> = Matrix::<f32>::zeros(3, 2);
/// let new_matrix: Matrix<f32> = &matrix + 2.0;
/// 
/// assert_eq!(new_matrix.get_data(), [2.0; 6])
/// ```
impl ops::Add<f32> for &Matrix<f32> {
    type Output = Matrix<f32>;

    fn add(self, rhs: f32) -> Matrix<f32> {
        let output_data: Vec<f32> = self.all_elements()
            .map(|val| val + rhs)
            .collect::<Vec<f32>>();

        return Matrix::new(output_data, self.get_rows(), self.get_cols()).unwrap();
    }
}

/// Add float to matrix.
/// 
/// ```
/// # use gpurs::linalg::Matrix;
/// let matrix: Matrix<f64> = Matrix::<f64>::zeros(3, 2);
/// let new_matrix: Matrix<f64> = &matrix + 2.0;
/// 
/// assert_eq!(new_matrix.get_data(), [2.0; 6]);
/// ```
impl ops::Add<f64> for &Matrix<f64> {
    type Output = Matrix<f64>;

    fn add(self, rhs: f64) -> Matrix<f64> {
        let output_data: Vec<f64> = self.all_elements()
            .map(|val| val + rhs)
            .collect::<Vec<f64>>();

            return Matrix::new(output_data, self.get_rows(), self.get_cols()).unwrap();
    }
}

/// Add matrix to matrix.
/// Matrices must be of the same size and shape.
/// 
/// ```
/// # use gpurs::{Result, linalg::Matrix};
/// # fn main() -> Result<()> {
/// let matrix_1: Matrix<f32> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2)?;
/// let matrix_2: Matrix<f32> = Matrix::<f32>::ones(2, 2);
/// 
/// let new_matrix: Matrix<f32> = (&matrix_1 + &matrix_2)?;
/// 
/// assert_eq!(new_matrix.get_data(), [2.0, 3.0, 4.0, 5.0]);
/// # Ok(())
/// # }
/// ```
impl ops::Add<&Matrix<f32>> for &Matrix<f32> {
    type Output = Result<Matrix<f32>>;

    fn add(self, rhs: &Matrix<f32>) -> Result<Matrix<f32>> {
        if (self.get_rows() != rhs.get_rows()) || (self.get_cols() != rhs.get_cols()) {
            return Err(Jeeperr::DimensionError)
        }

        let output_data: Vec<f32> = (0..self.get_elements()).into_iter()
            .map(|idx| self.lindex(idx) + rhs.lindex(idx))
            .collect::<Vec<f32>>();

        return Matrix::new(output_data, self.get_rows(), self.get_cols())
    }
}

/// Add matrix to matrix.
/// Matrices must be of the same size and shape.
/// 
/// ```
/// # use gpurs::{Result, linalg::Matrix};
/// # fn main() -> Result<()> {
/// let matrix_1: Matrix<f64> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2)?;
/// let matrix_2: Matrix<f64> = Matrix::<f64>::ones(2, 2);
/// 
/// let new_matrix: Matrix<f64> = (&matrix_1 + &matrix_2)?;
/// 
/// assert_eq!(new_matrix.get_data(), [2.0, 3.0, 4.0, 5.0]);
/// # Ok(())
/// # }
impl ops::Add<&Matrix<f64>> for &Matrix<f64> {
    type Output = Result<Matrix<f64>>;

    fn add(self, rhs: &Matrix<f64>) -> Result<Matrix<f64>> {
        if (self.get_rows() != rhs.get_rows()) || (self.get_cols() != rhs.get_cols()) {
            return Err(Jeeperr::DimensionError)
        }

        let output_data: Vec<f64> = (0..self.get_elements()).into_iter()
            .map(|idx| self.lindex(idx) + rhs.lindex(idx))
            .collect::<Vec<f64>>();

        return Matrix::new(output_data, self.get_rows(), self.get_cols())
    }
}

/// Add matrix to float.
/// 
/// ```
/// # use gpurs::linalg::Matrix;
/// let matrix: Matrix<f32> = Matrix::<f32>::ones(2, 3);
/// let new_matrix: Matrix<f32> = 1.0 + &matrix;
/// 
/// assert_eq!(new_matrix.get_data(), [2.0; 6]);
/// ```
impl ops::Add<&Matrix<f32>> for f32 {
    type Output = Matrix<f32>;

    fn add(self, rhs: &Matrix<f32>) -> Matrix<f32> {
        let output_data: Vec<f32> = rhs.all_elements()
            .map(|val| self + val)
            .collect::<Vec<f32>>();

        return Matrix::new(output_data, rhs.get_rows(), rhs.get_cols()).unwrap()
    }
}

/// Add matrix to float.
/// 
/// ```
/// # use gpurs::linalg::Matrix;
/// let matrix: Matrix<f64> = Matrix::<f64>::ones(2, 3);
/// let new_matrix: Matrix<f64> = 1.0 + &matrix;
/// 
/// assert_eq!(new_matrix.get_data(), [2.0; 6]);
/// ```
impl ops::Add<&Matrix<f64>> for f64 {
    type Output = Matrix<f64>;

    fn add(self, rhs: &Matrix<f64>) -> Matrix<f64> {
        let output_data: Vec<f64> = rhs.all_elements()
            .map(|val| self + val)
            .collect::<Vec<f64>>();

        return Matrix::new(output_data, rhs.get_rows(), rhs.get_cols()).unwrap()
    }
}

/// Negate matrix.
/// 
/// ```
/// # use gpurs::linalg::Matrix;
/// let matrix: Matrix<f32> = Matrix::<f32>::ones(4, 4);
/// let negated_matrix: Matrix<f32> = -&matrix;
/// 
/// assert_eq!(negated_matrix.get_data(), [-1.0; 16]);
/// ```
impl ops::Neg for &Matrix<f32> {
    type Output = Matrix<f32>;

    fn neg(self) -> Matrix<f32> {
        let output_data: Vec<f32> = self.all_elements()
            .map(|val| -val)
            .collect::<Vec<f32>>();

        return Matrix::new(output_data, self.get_rows(), self.get_cols()).unwrap()
    }
}

/// Negate matrix.
/// 
/// ```
/// # use gpurs::linalg::Matrix;
/// let matrix: Matrix<f64> = Matrix::<f64>::ones(4, 4);
/// let negated_matrix: Matrix<f64> = -&matrix;
/// 
/// assert_eq!(negated_matrix.get_data(), [-1.0; 16]);
/// ```
impl ops::Neg for &Matrix<f64> {
    type Output = Matrix<f64>;

    fn neg(self) -> Matrix<f64> {
        let output_data: Vec<f64> = self.all_elements()
            .map(|val| -val)
            .collect::<Vec<f64>>();

        return Matrix::new(output_data, self.get_rows(), self.get_cols()).unwrap()
    }
}

/// Subtract float from matrix.
/// 
/// ```
/// # use gpurs::linalg::Matrix;
/// let matrix: Matrix<f32> = Matrix::<f32>::zeros(3, 2);
/// let new_matrix: Matrix<f32> = &matrix - 2.0;
/// 
/// assert_eq!(new_matrix.get_data(), [-2.0; 6]);
/// ```
impl ops::Sub<f32> for &Matrix<f32> {
    type Output = Matrix<f32>;

    fn sub(self, rhs: f32) -> Matrix<f32> {
        let output_data: Vec<f32> = self.all_elements()
            .map(|val| val - rhs)
            .collect::<Vec<f32>>();

        return Matrix::new(output_data, self.get_rows(), self.get_cols()).unwrap()
    }
}

/// Subtract float from matrix.
/// 
/// ```
/// # use gpurs::linalg::Matrix;
/// let matrix: Matrix<f64> = Matrix::<f64>::zeros(3, 2);
/// let new_matrix: Matrix<f64> = &matrix - 2.0;
/// 
/// assert_eq!(new_matrix.get_data(), [-2.0; 6]);
/// ```
impl ops::Sub<f64> for &Matrix<f64> {
    type Output = Matrix<f64>;

    fn sub(self, rhs: f64) -> Matrix<f64> {
        let output_data: Vec<f64> = self.all_elements()
            .map(|val| val - rhs)
            .collect::<Vec<f64>>();

        return Matrix::new(output_data, self.get_rows(), self.get_cols()).unwrap()
    }
}

/// Subtract matrix from matrix.
/// Matrices must be of the same size and shape.
/// 
/// ```
/// # use gpurs::{Result, linalg::Matrix};
/// # fn main() -> Result<()> {
/// let matrix_1: Matrix<f32> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2)?;
/// let matrix_2: Matrix<f32> = Matrix::<f32>::ones(2, 2);
/// 
/// let new_matrix: Matrix<f32> = (&matrix_1 - &matrix_2)?;
/// 
/// assert_eq!(new_matrix.get_data(), [0.0, 1.0, 2.0, 3.0]);
/// # Ok(())
/// # }
/// ```
impl ops::Sub<&Matrix<f32>> for &Matrix<f32> {
    type Output = Result<Matrix<f32>>;

    fn sub(self, rhs: &Matrix<f32>) -> Result<Matrix<f32>> {
        if (self.get_rows() != rhs.get_rows()) || (self.get_cols() != rhs.get_cols()) {
            return Err(Jeeperr::DimensionError)
        }

        let output_data: Vec<f32> = (0..self.get_elements()).into_iter()
            .map(|idx| self.lindex(idx) - rhs.lindex(idx))
            .collect::<Vec<f32>>();

        return Matrix::new(output_data, self.get_rows(), self.get_cols())
    }
}

/// Subtract matrix from matrix.
/// Matrices must be of the same size and shape.
/// 
/// ```
/// # use gpurs::{Result, linalg::Matrix};
/// # fn main() -> Result<()> {
/// let matrix_1: Matrix<f64> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2)?;
/// let matrix_2: Matrix<f64> = Matrix::<f64>::ones(2, 2);
/// 
/// let new_matrix: Matrix<f64> = (&matrix_1 - &matrix_2)?;
/// 
/// assert_eq!(new_matrix.get_data(), [0.0, 1.0, 2.0, 3.0]);
/// # Ok(())
/// # }
impl ops::Sub<&Matrix<f64>> for &Matrix<f64> {
    type Output = Result<Matrix<f64>>;

    fn sub(self, rhs: &Matrix<f64>) -> Result<Matrix<f64>> {
        if (self.get_rows() != rhs.get_rows()) || (self.get_cols() != rhs.get_cols()) {
            return Err(Jeeperr::DimensionError)
        }

        let output_data: Vec<f64> = (0..self.get_elements()).into_iter()
            .map(|idx| self.lindex(idx) - rhs.lindex(idx))
            .collect::<Vec<f64>>();

        return Matrix::new(output_data, self.get_rows(), self.get_cols())
    }
}

/// Subtract matrix from float.
/// 
/// ```
/// # use gpurs::linalg::Matrix;
/// let matrix: Matrix<f32> = Matrix::<f32>::ones(2, 3);
/// let new_matrix: Matrix<f32> = 1.0 - &matrix;
/// 
/// assert_eq!(new_matrix.get_data(), [0.0; 6]);
/// ```
impl ops::Sub<&Matrix<f32>> for f32 {
    type Output = Matrix<f32>;

    fn sub(self, rhs: &Matrix<f32>) -> Matrix<f32> {
        let output_data: Vec<f32> = rhs.all_elements()
            .map(|val| self - val)
            .collect::<Vec<f32>>();

        return Matrix::new(output_data, rhs.get_rows(), rhs.get_cols()).unwrap();
    }
}

/// Subtract matrix from float.
/// 
/// ```
/// # use gpurs::linalg::Matrix;
/// let matrix: Matrix<f64> = Matrix::<f64>::ones(2, 3);
/// let new_matrix: Matrix<f64> = 1.0 - &matrix;
/// 
/// assert_eq!(new_matrix.get_data(), [0.0; 6]);
/// ```
impl ops::Sub<&Matrix<f64>> for f64 {
    type Output = Matrix<f64>;

    fn sub(self, rhs: &Matrix<f64>) -> Matrix<f64> {
        let output_data: Vec<f64> = rhs.all_elements()
            .map(|val| self - val)
            .collect::<Vec<f64>>();

        return Matrix::new(output_data, rhs.get_rows(), rhs.get_cols()).unwrap()
    }
}

/// Multiply matrix by float.
/// 
/// ```
/// # use gpurs::linalg::Matrix;
/// let matrix: Matrix<f32> = Matrix::<f32>::ones(4, 3);
/// let new_matrix: Matrix<f32> = &matrix * 5.0;
/// 
/// assert_eq!(new_matrix.get_data(), [5.0; 12]);
/// ```
impl ops::Mul<f32> for &Matrix<f32> {
    type Output = Matrix<f32>;

    fn mul(self, rhs: f32) -> Matrix<f32> {
        let output_data: Vec<f32> = self.all_elements()
            .map(|val| val * rhs)
            .collect::<Vec<f32>>();

        return Matrix::new(output_data, self.get_rows(), self.get_cols()).unwrap()
    }
}

/// Multiply matrix by float.
/// 
/// ```
/// # use gpurs::linalg::Matrix;
/// let matrix: Matrix<f64> = Matrix::<f64>::ones(4, 3);
/// let new_matrix: Matrix<f64> = &matrix * 5.0;
/// 
/// assert_eq!(new_matrix.get_data(), [5.0; 12]);
/// ```
impl ops::Mul<f64> for &Matrix<f64> {
    type Output = Matrix<f64>;

    fn mul(self, rhs: f64) -> Matrix<f64> {
        let output_data: Vec<f64> = self.all_elements()
            .map(|val| val * rhs)
            .collect::<Vec<f64>>();

        return Matrix::new(output_data, self.get_rows(), self.get_cols()).unwrap()
    }
}

/// Multiply matrix by matrix.
/// 
/// Number of columns in the left matrix must equal the number of rows in the right matrix.
/// The shape of the resultant matrix will be (left rows) by (right cols)
/// 
/// The value at every [i, j] index of the resultant matrix is equal to the dot product of the
/// i-th row of the left matrix and the j-th col of the right matrix.
/// 
/// ```
/// # use gpurs::{Result, linalg::Matrix};
/// # fn main() -> Result<()> {
/// let left_matrix: Matrix<f32> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2)?;
/// let right_matrix: Matrix<f32> = Matrix::new(vec![6.0, 1.0, 5.0, 2.0, 4.0, 3.0], 2, 3)?;
/// 
/// let product: Matrix<f32> = (&left_matrix * &right_matrix)?;
/// 
/// assert_eq!(product.get_data(), [10.0, 9.0, 11.0, 26.0, 19.0, 27.0]);
/// # Ok(())
/// # }
/// ```
impl ops::Mul<&Matrix<f32>> for &Matrix<f32> {
    type Output = Result<Matrix<f32>>;

    fn mul(self, rhs: &Matrix<f32>) -> Result<Matrix<f32>> {
        if self.get_cols() != rhs.get_rows() {
            return Err(Jeeperr::DimensionError)
        }
        let new_n_elements: usize = self.get_rows() * rhs.get_cols();
        let mut output_data: Vec<f32> = Vec::with_capacity(new_n_elements);

        for lhs_row in self.all_rows() {
            for rhs_col in rhs.all_cols() {
                output_data.push(lhs_row.ref_sumproduct(&rhs_col)?);
            }
         }

        return Matrix::new(output_data, self.get_rows(), rhs.get_cols())
    }
}

/// Multiply matrix by matrix.
/// 
/// Number of columns in the left matrix must equal the number of rows in the right matrix.
/// The shape of the resultant matrix will be (left rows) by (right cols)
/// 
/// The value at every [i, j] index of the resultant matrix is equal to the dot product of the
/// i-th row of the left matrix and the j-th col of the right matrix.
/// 
/// ```
/// # use gpurs::{Result, linalg::Matrix};
/// # fn main() -> Result<()> {
/// let left_matrix: Matrix<f64> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2)?;
/// let right_matrix: Matrix<f64> = Matrix::new(vec![6.0, 1.0, 5.0, 2.0, 4.0, 3.0], 2, 3)?;
/// 
/// let product: Matrix<f64> = (&left_matrix * &right_matrix)?;
/// 
/// assert_eq!(product.get_data(), [10.0, 9.0, 11.0, 26.0, 19.0, 27.0]);
/// # Ok(())
/// # }
/// ```
impl ops::Mul<&Matrix<f64>> for &Matrix<f64> {
    type Output = Result<Matrix<f64>>;

    fn mul(self, rhs: &Matrix<f64>) -> Result<Matrix<f64>> {
        if self.get_cols() != rhs.get_rows() {
            return Err(Jeeperr::DimensionError)
        }
        let new_n_elements: usize = self.get_rows() * rhs.get_cols();
        let mut output_data: Vec<f64> = Vec::with_capacity(new_n_elements);

        for lhs_row in self.all_rows() {
            for rhs_col in rhs.all_cols() {
                output_data.push(lhs_row.ref_sumproduct(&rhs_col)?);
            }
         }

        return Matrix::new(output_data, self.get_rows(), rhs.get_cols())
    }
}

/// Multiply float by matrix.
/// 
/// ```
/// # use gpurs::linalg::Matrix;
/// let matrix: Matrix<f32> = Matrix::<f32>::ones(2, 5);
/// let new_matrix: Matrix<f32> = 8.0 * &matrix;
/// 
/// assert_eq!(new_matrix.get_data(), [8.0; 10]);
/// ```
impl ops::Mul<&Matrix<f32>> for f32 {
    type Output = Matrix<f32>;

    fn mul(self, rhs: &Matrix<f32>) -> Matrix<f32> {
        let output_data = rhs.all_elements()
            .map(|val| self * val)
            .collect::<Vec<f32>>();

        return Matrix::new(output_data, rhs.get_rows(), rhs.get_cols()).unwrap()
    }
}

/// Multiply float by matrix.
/// 
/// ```
/// # use gpurs::linalg::Matrix;
/// let matrix: Matrix<f64> = Matrix::<f64>::ones(2, 5);
/// let new_matrix: Matrix<f64> = 8.0 * &matrix;
/// 
/// assert_eq!(new_matrix.get_data(), [8.0; 10]);
/// ```
impl ops::Mul<&Matrix<f64>> for f64 {
    type Output = Matrix<f64>;

    fn mul(self, rhs: &Matrix<f64>) -> Matrix<f64> {
        let output_data = rhs.all_elements()
            .map(|val| self * val)
            .collect::<Vec<f64>>();

        return Matrix::new(output_data, rhs.get_rows(), rhs.get_cols()).unwrap()
    }
}

/// Divide matrix by float.
/// 
/// ```
/// # use gpurs::linalg::Matrix;
/// let matrix: Matrix<f32> = Matrix::<f32>::ones(2, 5);
/// let new_matrix: Matrix<f32> = &matrix / 2.0;
/// 
/// assert_eq!(new_matrix.get_data(), [0.5; 10]);
/// ```
impl ops::Div<f32> for &Matrix<f32> {
    type Output = Matrix<f32>;

    fn div(self, rhs: f32) -> Matrix<f32> {
        let output_data: Vec<f32> = self.all_elements()
            .map(|val| val / rhs)
            .collect::<Vec<f32>>();

        return Matrix::new(output_data, self.get_rows(), self.get_cols()).unwrap()
    }
}

/// Divide matrix by float.
/// 
/// ```
/// # use gpurs::linalg::Matrix;
/// let matrix: Matrix<f64> = Matrix::<f64>::ones(2, 5);
/// let new_matrix: Matrix<f64> = &matrix / 2.0;
/// 
/// assert_eq!(new_matrix.get_data(), [0.5; 10]);
/// ```
impl ops::Div<f64> for &Matrix<f64> {
    type Output = Matrix<f64>;

    fn div(self, rhs: f64) -> Matrix<f64> {
        let output_data: Vec<f64> = self.all_elements()
            .map(|val| val / rhs)
            .collect::<Vec<f64>>();

        return Matrix::new(output_data, self.get_rows(), self.get_cols()).unwrap()
    }
}

/// Divide float by matrix.
/// 
/// ```
/// # use gpurs::linalg::Matrix;
/// let matrix: Matrix<f32> = Matrix::<f32>::ones(2, 5);
/// let new_matrix: Matrix<f32> = 2.0 / &matrix;
/// 
/// assert_eq!(new_matrix.get_data(), [2.0; 10])
/// ```
impl ops::Div<&Matrix<f32>> for f32 {
    type Output = Matrix<f32>;

    fn div(self, rhs: &Matrix<f32>) -> Matrix<f32> {
        let output_data: Vec<f32> = rhs.all_elements()
            .map(|val| self / val)
            .collect::<Vec<f32>>();

        return Matrix::new(output_data, rhs.get_rows(), rhs.get_cols()).unwrap()
    }
}

/// Divide float by matrix.
/// 
/// ```
/// # use gpurs::linalg::Matrix;
/// let matrix: Matrix<f64> = Matrix::<f64>::ones(2, 5);
/// let new_matrix: Matrix<f64> = 2.0 / &matrix;
/// 
/// assert_eq!(new_matrix.get_data(), [2.0; 10])
/// ```
impl ops::Div<&Matrix<f64>> for f64 {
    type Output = Matrix<f64>;

    fn div(self, rhs: &Matrix<f64>) -> Matrix<f64> {
        let output_data: Vec<f64> = rhs.all_elements()
            .map(|val| self / val)
            .collect::<Vec<f64>>();

            return Matrix::new(output_data, rhs.get_rows(), rhs.get_cols()).unwrap()
    }
}