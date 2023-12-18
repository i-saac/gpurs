//! Matrix object and implementations

use std::ops;

use crate::IsFloat;
use crate::Result;
use crate::Jeeperr;

/// Generic dynamically-sized Matrix struct for the entire gpurs library.
#[derive(Debug, Clone)]
pub struct Matrix<T: IsFloat + std::fmt::Debug + Copy + Clone> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: IsFloat + std::fmt::Debug + Copy + Clone> Matrix<T> {
    /// Create new matrix (includes error checking).
    /// 
    /// ```
    /// # use gpurs::{Result, linalg::Matrix};
    /// # fn main() -> Result<()> {
    /// // This code will run, because rows * cols
    /// // equals the number of elements in the data vector
    /// let new_matrix: Matrix<f32> = Matrix::new(vec![0.0; 4], 2, 2)?;
    /// # Ok(())
    /// # }
    /// ```
    /// 
    /// ```should_panic
    /// # use gpurs::{Result, linalg::Matrix};
    /// # fn main() -> Result<()> {
    /// // This code will panic, because rows * cols
    /// // does not equal the number of elements in the data vector
    /// let new_matrix: Matrix<f32> = Matrix::new(vec![0.0; 4], 3, 3)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(data: Vec<T>, rows: usize, cols: usize) -> Result<Matrix<T>> {
        let vec_len: usize = data.len();
        let comp_len: usize = rows * cols;

        if vec_len == comp_len {
            let output: Matrix<T> = Matrix { data, rows, cols };

            return Ok(output)
        }
        else {
            return Err(Jeeperr::DimensionError)
        }
    }

    /// Get number of rows in matrix.
    /// 
    /// ```
    /// # use gpurs::{Result, linalg::Matrix};
    /// # fn main() -> Result<()> {
    /// let new_matrix: Matrix<f64> = Matrix::new(vec![0.0; 6], 3, 2)?;
    /// 
    /// assert_eq!(new_matrix.get_rows(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_rows(&self) -> usize {
        return self.rows
    }

    /// Return iterator object full of every row in the matrix.
    /// 
    /// ```
    /// # use gpurs::{Result, linalg::Matrix};
    /// # fn main() -> Result<()> {
    /// let new_matrix: Matrix<f32> = Matrix::new(vec![0.0; 6], 3, 2)?;
    /// for row in new_matrix.all_rows() {
    ///     // Do something neat
    /// }
    /// # Ok(())
    /// # }
    /// ```
    /// 
    /// This does not return a mutable reference,
    /// so changing the row matrix spat out by the iterator
    /// does not change the underlying matrix.
    pub fn all_rows(&self) -> RowMatIter<T> {
        return RowMatIter { row: 0, mat: self.clone() }
    }

    /// Get number of cols in matrix.
    /// 
    /// ```
    /// # use gpurs::{Result, linalg::Matrix};
    /// # fn main() -> Result<()> {
    /// let new_matrix: Matrix<f64> = Matrix::new(vec![0.0; 6], 3, 2)?;
    /// 
    /// assert_eq!(new_matrix.get_cols(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_cols(&self) -> usize {
        return self.cols
    }

    /// Return iterator object full of every column in the matrix.
    /// 
    /// ```
    /// # use gpurs::{Result, linalg::Matrix};
    /// # fn main() -> Result<()> {
    /// let new_matrix: Matrix<f32> = Matrix::new(vec![0.0; 6], 3, 2)?;
    /// for col in new_matrix.all_cols() {
    ///     // Do something neat
    /// }
    /// # Ok(())
    /// # }
    /// ```
    /// 
    /// This does not return a mutable reference,
    /// so changing the column matrix spat out by the iterator
    /// does not change the underlying matrix.
    pub fn all_cols(&self) -> ColMatIter<T> {
        return ColMatIter { col: 0, mat: self.clone() }
    }

    /// Get number of elements in matrix.
    /// 
    /// ```
    /// # use gpurs::{Result, linalg::Matrix};
    /// # fn main() -> Result<()> {
    /// let new_matrix: Matrix<f64> = Matrix::new(vec![0.0; 6], 3, 2)?;
    /// 
    /// assert_eq!(new_matrix.get_elements(), 6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_elements(&self) -> usize {
        return self.rows * self.cols
    }

    /// Return iterator object full of every element in the matrix.
    /// 
    /// ```
    /// # use gpurs::{Result, linalg::Matrix};
    /// # fn main() -> Result<()> {
    /// let new_matrix: Matrix<f32> = Matrix::new(vec![0.0; 6], 3, 2)?;
    /// for element in new_matrix.all_elements() {
    ///     // Do something neat
    /// }
    /// # Ok(())
    /// # }
    pub fn all_elements(&self) -> MatIter<T> {
        return MatIter { index: 0, mat: self.clone() }
    }

    /// Get matrix data in slice form.
    /// 
    /// ```
    /// # use gpurs::{Result, linalg::Matrix};
    /// # fn main() -> Result<()> {
    /// let new_matrix: Matrix<f64> = Matrix::new(vec![0.0; 6], 3, 2)?;
    /// 
    /// assert_eq!(new_matrix.get_data(), [0.0; 6]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_data(&self) -> &[T] {
        return &self.data
    }

    /// Get indexed row of matrix in matrix form.
    /// 
    /// ```
    /// # use gpurs::{Result, linalg::Matrix};
    /// # fn main() -> Result<()> {
    /// let new_matrix: Matrix<f32> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2)?;
    /// let second_row: Matrix<f32> = Matrix::new(vec![3.0, 4.0], 1, 2)?;
    /// 
    /// assert_eq!(new_matrix.row_matrix(1)?.get_data(), second_row.get_data());
    /// # Ok(())
    /// # }
    /// ```
    pub fn row_matrix(&self, row_idx: usize) -> Result<Matrix<T>> {
        if row_idx >= self.rows {
            return Err(Jeeperr::IndexError)
        }

        let lower_data_idx: usize = row_idx * self.cols;
        let upper_data_idx: usize = (row_idx + 1) * self.cols;

        let output_row_data: Vec<T> = self.data[lower_data_idx..upper_data_idx].to_vec();

        return Ok(Matrix { data: output_row_data, rows: 1, cols: self.cols })
    }

    /// Get indexed row of matrix in vector form.
    /// 
    /// ```
    /// # use gpurs::{Result, linalg::Matrix};
    /// # fn main() -> Result<()> {
    /// let new_matrix: Matrix<f64> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2)?;
    /// 
    /// assert_eq!(new_matrix.row_vec(0)?, vec![1.0, 2.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn row_vec(&self, row_idx: usize) -> Result<Vec<T>> {
        if row_idx >= self.rows {
            return Err(Jeeperr::IndexError)
        }

        let lower_data_idx: usize = row_idx * self.cols;
        let upper_data_idx: usize = (row_idx + 1) * self.cols;

        let output_row: Vec<T> = self.data[lower_data_idx..upper_data_idx].to_vec();

        return Ok(output_row)
    }

    /// Get indexed col of matrix in matrix form.
    /// 
    /// ```
    /// # use gpurs::{Result, linalg::Matrix};
    /// # fn main() -> Result<()> {
    /// let new_matrix: Matrix<f32> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2)?;
    /// let second_col: Matrix<f32> = Matrix::new(vec![2.0, 4.0], 2, 1)?;
    /// 
    /// assert_eq!(new_matrix.col_matrix(1)?.get_data(), second_col.get_data());
    /// # Ok(())
    /// # }
    pub fn col_matrix(&self, col_idx: usize) -> Result<Matrix<T>> {
        if col_idx >= self.cols {
            return Err(Jeeperr::IndexError)
        }

        let lower_data_idx: usize = col_idx;
        let upper_data_idx: usize = (self.rows - 1) * self.cols + col_idx;

        let output_col_data: Vec<T> = (lower_data_idx..=upper_data_idx).step_by(self.cols)
            .map(|idx| self.data[idx])
            .collect::<Vec<T>>();

        return Ok(Matrix { data: output_col_data, rows: self.rows, cols: 1 })
    }

    /// Get indexed col of matrix in vector form.
    /// 
    /// ```
    /// # use gpurs::{Result, linalg::Matrix};
    /// # fn main() -> Result<()> {
    /// let new_matrix: Matrix<f64> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2)?;
    /// 
    /// assert_eq!(new_matrix.col_vec(0)?, vec![1.0, 3.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn col_vec(&self, col_idx: usize) -> Result<Vec<T>> {
        if col_idx >= self.cols {
            return Err(Jeeperr::IndexError)
        }

        let lower_data_idx: usize = col_idx;
        let upper_data_idx: usize = self.rows * self.cols;

        let output_col: Vec<T> = (lower_data_idx..upper_data_idx).step_by(self.cols)
            .map(|idx| self.data[idx])
            .collect::<Vec<T>>();

        return Ok(output_col)
    }

    /// Index matrix using usize slices, outputing matrix of the index intersections.
    /// 
    /// Think of drawing lines across each selected row and down each selected column,
    /// then compiling the elements at their intersections into a new matrix.
    /// 
    /// ```
    /// # use gpurs::{Result, linalg::Matrix};
    /// # fn main() -> Result<()> {
    /// let matrix_data: Vec<f32> = (1..=9).into_iter().map(|x| x as f32).collect::<Vec<f32>>();
    /// let new_matrix: Matrix<f32> = Matrix::new(matrix_data, 3, 3)?;
    /// let sliced_matrix: Matrix<f32> = new_matrix.slice_index(&[0, 1], &[0, 2]);
    /// 
    /// assert_eq!(sliced_matrix.get_data(), [1.0, 3.0, 4.0, 6.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn slice_index(&self, row_idcs: &[usize], col_idcs: &[usize]) -> Matrix<T> {
        let output_rows: usize = row_idcs.len();
        let output_cols: usize = col_idcs.len();
        let mut output_data: Vec<T> = Vec::with_capacity(output_rows * output_cols);
        for row_idx in row_idcs {
            for col_idx in col_idcs {
                output_data.push(self.data[row_idx * self.cols + col_idx]);
            }
        }

        return Matrix { data: output_data, rows: output_rows, cols: output_cols }
    }

    /// Matrix transpose
    /// 
    /// ```
    /// # use gpurs::{Result, linalg::Matrix};
    /// # fn main() -> Result<()> {
    /// let new_matrix: Matrix<f64> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2)?;
    /// 
    /// assert_eq!(new_matrix.transpose().get_data(), [1.0, 3.0, 2.0, 4.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn transpose(&self) -> Matrix<T> {
        let mut transpose_data: Vec<T> = Vec::with_capacity(self.rows * self.cols);

        for transpose_row in 0..self.cols {
            for transpose_col in 0..self.rows {
                transpose_data.push(self.data[transpose_col * self.cols + transpose_row]);
            }
        }

        Matrix { data: transpose_data, rows: self.cols, cols: self.rows }
    }

    /// Returns indexed value of flattened matrix.
    /// Shorthand for matrix.get_data()\[linear_index\].
    pub fn lindex(&self, linear_index: usize) -> T {
        self.data[linear_index]
    }
}

impl Matrix<f32> {
    /// Create new matrix of zeros.
    /// 
    /// ```
    /// # use gpurs::linalg::Matrix;
    /// let zero_matrix: Matrix<f32> = Matrix::<f32>::zeros(2, 3);
    /// 
    /// assert_eq!(zero_matrix.get_data(), [0.0; 6]);
    /// ```
    pub fn zeros(rows: usize, cols: usize) -> Matrix<f32> {
        let data: Vec<f32> = vec![0.0; rows * cols];

        return Matrix { data, rows, cols }
    }

    /// Create new matrix of ones.
    /// 
    /// ```
    /// # use gpurs::linalg::Matrix;
    /// let ones_matrix: Matrix<f32> = Matrix::<f32>::ones(2, 3);
    /// 
    /// assert_eq!(ones_matrix.get_data(), [1.0; 6]);
    /// ```
    pub fn ones(rows: usize, cols: usize) -> Matrix<f32> {
        let data: Vec<f32> = vec![1.0; rows * cols];

        return Matrix { data, rows, cols }
    }

    /// Create new identity matrix.
    /// 
    /// ```
    /// # use gpurs::linalg::Matrix;
    /// let identity_matrix: Matrix<f32> = Matrix::<f32>::identity(2);
    /// 
    /// assert_eq!(identity_matrix.get_data(), [1.0, 0.0, 0.0, 1.0]);
    /// ```
    pub fn identity(dim_n: usize) -> Matrix<f32> {
        let mut data: Vec<f32> = vec![0.0; dim_n * dim_n];
        
        for diagonal_idx in 0..dim_n {
            data[diagonal_idx * dim_n + diagonal_idx] = 1.0;
        }

        return Matrix { data, rows: dim_n, cols: dim_n };
    }
}

impl Matrix<f64> {
    /// Create new matrix of zeros.
    /// 
    /// ```
    /// # use gpurs::linalg::Matrix;
    /// let zero_matrix: Matrix<f64> = Matrix::<f64>::zeros(2, 3);
    /// 
    /// assert_eq!(zero_matrix.get_data(), [0.0; 6]);
    /// ```
    pub fn zeros(rows: usize, cols: usize) -> Matrix<f64> {
        let data: Vec<f64> = vec![0.0; rows * cols];

        return Matrix { data, rows, cols }
    }

    /// Create new matrix of ones.
    /// 
    /// ```
    /// # use gpurs::linalg::Matrix;
    /// let ones_matrix: Matrix<f64> = Matrix::<f64>::ones(2, 3);
    /// 
    /// assert_eq!(ones_matrix.get_data(), [1.0; 6]);
    /// ```
    pub fn ones(rows: usize, cols: usize) -> Matrix<f64> {
        let data: Vec<f64> = vec![1.0; rows * cols];

        return Matrix { data, rows, cols }
    }

    /// Create new identity matrix.
    /// 
    /// ```
    /// # use gpurs::linalg::Matrix;
    /// let identity_matrix: Matrix<f64> = Matrix::<f64>::identity(2);
    /// 
    /// assert_eq!(identity_matrix.get_data(), [1.0, 0.0, 0.0, 1.0]);
    /// ```
    pub fn identity(dim_n: usize) -> Matrix<f64> {
        let mut data: Vec<f64> = vec![0.0; dim_n * dim_n];
        
        for diagonal_idx in 0..dim_n {
            data[diagonal_idx * dim_n + diagonal_idx] = 1.0;
        }

        return Matrix { data, rows: dim_n, cols: dim_n };
    }
}

/// Get value of matrix at index [row, col].
/// 
/// ```
/// # use gpurs::{Result, linalg::Matrix};
/// # fn main() -> Result<()> {
/// # let matrix: Matrix<f32> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2)?;
/// let row_idx: usize = 1;
/// let col_idx: usize = 0;
/// 
/// let indexed_value: f32 = matrix[[row_idx, col_idx]];
/// # assert_eq!(indexed_value, 3.0);
/// # Ok(())
/// # }
/// ```
impl<T: IsFloat + std::fmt::Debug + Copy + Clone> ops::Index<[usize; 2]> for Matrix<T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &T {
        return &self.data[idx[0] * self.cols + idx[1]]
    }
}

/// Modify value of matrix at index [row, col].
/// 
/// ```
/// # use gpurs::{Result, linalg::Matrix};
/// # fn main() -> Result<()> {
/// let mut matrix: Matrix<f64> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2)?;
/// 
/// let row_idx: usize = 0;
/// let col_idx: usize = 1;
/// 
/// let new_value: f64 = 5.0;
/// 
/// matrix[[row_idx, col_idx]] = new_value;
/// 
/// assert_eq!(matrix.get_data(), [1.0, 5.0, 3.0, 4.0]);
/// # Ok(())
/// # }
/// ```
impl<T: IsFloat + std::fmt::Debug + Copy + Clone> ops::IndexMut<[usize; 2]> for Matrix<T> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut T {
        return &mut self.data[index[0] * self.cols + index[1]];
    }
}

/// Print matrix in a readable way.
///
/// Assume you have this matrix:
/// 
/// ```ignore
/// let matrix: Matrix<f32> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)?;
/// ```
/// 
/// Printing using the debug logger as shown below would result in the following output:
/// 
/// ```ignore
/// println!("{:?}", matrix);
/// 
/// // This prints as
/// // Matrix { data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], rows: 2, cols: 3 }
/// ```
/// 
/// This implementation allows for standard printing which results in the following output:
/// 
/// ```ignore
/// println!("{}", matrix);
/// 
/// // This prints as
/// // [[1.0, 2.0, 3.0]
/// //  [4.0, 5.0, 6.0]]
/// ```
impl<T: IsFloat + std::fmt::Debug + Copy + Clone> std::fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "[{:?}", self.row_vec(0).unwrap())?;
        if self.rows > 1 {
            writeln!(f, "")?;
            for row_idx in 1..self.rows {
                write!(f, " {:?}", self.row_vec(row_idx).unwrap())?;
                if row_idx < self.rows - 1 {
                    write!(f, "\n")?;
                }
            }
        }
        writeln!(f, "]")?;

        Ok(())
    }
}

/// Iterator object that returns a row of the input matrix.
/// Constructed by matrix.all_rows(), cannot be manually constructed.
pub struct RowMatIter<T: IsFloat + std::fmt::Debug + Copy + Clone> {
    row: usize,
    mat: Matrix<T>
}

/// Iterator implementation for RowMatIter
impl<T: IsFloat + std::fmt::Debug + Copy + Clone> Iterator for RowMatIter<T> {
    type Item = Matrix<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.row < self.mat.get_rows() {
            self.row += 1;

            return Some(self.mat.row_matrix(self.row - 1).unwrap());
        }
        
        return None;
    }
}

/// Iterator object that returns a column of the input matrix.
/// Constructed by matrix.all_cols(), cannot be manually constructed.
pub struct ColMatIter<T: IsFloat + std::fmt::Debug + Copy + Clone> {
    col: usize,
    mat: Matrix<T>
}

/// Iterator implementation for ColMatIter
impl<T: IsFloat + std::fmt::Debug + Copy + Clone> Iterator for ColMatIter<T> {
    type Item = Matrix<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.col < self.mat.get_cols() {
            self.col += 1;

            return Some(self.mat.col_matrix(self.col - 1).unwrap());
        }
        
        return None;
    }
}

/// Iterator object that returns an element of the input matrix.
/// Constructed by matrix.all_elements(), cannot be manually constructed.
pub struct MatIter<T: IsFloat + std::fmt::Debug + Copy + Clone> {
    index: usize,
    mat: Matrix<T>
}

/// Iterator Implementation for MatIter
impl<T: IsFloat + std::fmt::Debug + Copy + Clone> Iterator for MatIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let n_elements: usize = self.mat.get_rows() * self.mat.get_cols();
        if self.index < n_elements {
            self.index += 1;

            return Some(self.mat.lindex(self.index - 1));
        }

        return None;
    }
}