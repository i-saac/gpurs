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
    cols: usize
}

impl<T: IsFloat + std::fmt::Debug + Copy + Clone> Matrix<T> {
    /// Create new matrix (includes error checking).
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
    pub fn get_rows(&self) -> usize {
        return self.rows
    }

    pub fn all_rows(&self) -> RowMatIter<T> {
        return RowMatIter { row: 0, mat: self.clone() }
    }

    /// Get number of cols in matrix.
    pub fn get_cols(&self) -> usize {
        return self.cols
    }

    pub fn all_cols(&self) -> ColMatIter<T> {
        return ColMatIter { col: 0, mat: self.clone() }
    }

    /// Get matrix data in slice form.
    pub fn get_data(&self) -> &[T] {
        return &self.data
    }

    /// Get indexed row of matrix in matrix form.
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

    /// Matrix transpose.
    pub fn transpose(self) -> Matrix<T> {
        let mut transpose_data: Vec<T> = Vec::with_capacity(self.rows * self.cols);

        for transpose_row in 0..self.cols {
            for transpose_col in 0..self.rows {
                transpose_data.push(self.data[transpose_col * self.cols + transpose_row]);
            }
        }

        Matrix { data: transpose_data, rows: self.cols, cols: self.rows }
    }
}

impl Matrix<f32> {
    /// Create new matrix of zeros.
    pub fn zeros(rows: usize, cols: usize) -> Matrix<f32> {
        let data: Vec<f32> = vec![0.0; rows * cols];

        return Matrix { data, rows, cols }
    }

    /// Create new matrix of ones.
    pub fn ones(rows: usize, cols: usize) -> Matrix<f32> {
        let data: Vec<f32> = vec![1.0; rows * cols];

        return Matrix { data, rows, cols }
    }

    /// Create new identity matrix.
    pub fn identity(dim_n: usize) -> Matrix<f32> {
        let mut data: Vec<f32> = vec![0.0; dim_n * dim_n];
        
        for diagonal_idx in 0..dim_n {
            data[diagonal_idx * dim_n + diagonal_idx] = 1.0;
        }

        return Matrix { data, rows: dim_n, cols: dim_n };
    }

    /// Elementwise multiplication between two matrices.
    pub fn elementwise(self, rhs: Matrix<f32>) -> Result<Matrix<f32>> {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = self.rows * self.cols;
        let mut ewmult_data: Vec<f32> = Vec::with_capacity(n_elements);

        for element in 0..n_elements {
            ewmult_data.push(self.data[element] * rhs.data[element]);
        }

        Ok(Matrix { data: ewmult_data, rows: self.rows, cols: self.cols })
    }
}

impl Matrix<f64> {
    /// Create new matrix of zeros.
    pub fn zeros(rows: usize, cols: usize) -> Matrix<f64> {
        let data: Vec<f64> = vec![0.0; rows * cols];

        return Matrix { data, rows, cols }
    }

    /// Create new matrix of ones.
    pub fn ones(rows: usize, cols: usize) -> Matrix<f64> {
        let data: Vec<f64> = vec![1.0; rows * cols];

        return Matrix { data, rows, cols }
    }

    /// Create new identity matrix.
    pub fn identity(dim_n: usize) -> Matrix<f64> {
        let mut data: Vec<f64> = vec![0.0; dim_n * dim_n];
        
        for diagonal_idx in 0..dim_n {
            data[diagonal_idx * dim_n + diagonal_idx] = 1.0;
        }

        return Matrix { data, rows: dim_n, cols: dim_n };
    }

    /// Elementwise multiplication between two matrices.
    pub fn elementwise(self, rhs: Matrix<f64>) -> Result<Matrix<f64>> {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = self.rows * self.cols;
        let mut ewmult_data: Vec<f64> = Vec::with_capacity(n_elements);

        for element in 0..n_elements {
            ewmult_data.push(self.data[element] * rhs.data[element]);
        }

        Ok(Matrix { data: ewmult_data, rows: self.rows, cols: self.cols })
    }
}

/// Get value of matrix at index [row, col].
impl<T: IsFloat + std::fmt::Debug + Copy + Clone> ops::Index<[usize; 2]> for Matrix<T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &T {
        return &self.data[idx[0] * self.cols + idx[1]]
    }
}

/// Modify value of matrix at index [row, col].
impl<T: IsFloat + std::fmt::Debug + Copy + Clone> ops::IndexMut<[usize; 2]> for Matrix<T> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut T {
        return &mut self.data[index[0] * self.cols + index[1]];
    }
}

/// Add matrix with float.
impl ops::Add<f32> for Matrix<f32> {
    type Output = Matrix<f32>;

    fn add(self, rhs: f32) -> Matrix<f32> {
        let output_data: Vec<f32> = self.data.iter()
            .map(|val| val + rhs)
            .collect::<Vec<f32>>();

        Matrix { data: output_data, rows: self.rows, cols: self.cols }
    }
}

/// Add matrix with float.
impl ops::Add<f64> for Matrix<f64> {
    type Output = Matrix<f64>;

    fn add(self, rhs: f64) -> Matrix<f64> {
        let output_data: Vec<f64> = self.data.iter()
            .map(|val| val + rhs)
            .collect::<Vec<f64>>();

        Matrix { data: output_data, rows: self.rows, cols: self.cols }
    }
}

/// Add Matrix with Matrix.
impl ops::Add<Matrix<f32>> for Matrix<f32> {
    type Output = Result<Matrix<f32>>;

    fn add(self, rhs: Matrix<f32>) -> Result<Matrix<f32>> {
        if (self.rows != rhs.rows) || (self.cols != rhs.cols) {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = self.rows * self.cols;
        let output_data: Vec<f32> = (0..n_elements).into_iter()
            .map(|idx| self.data[idx] + rhs.data[idx])
            .collect::<Vec<f32>>();

        Ok(Matrix { data: output_data, rows: self.rows, cols: self.cols })
    }
}

/// Add Matrix with Matrix.
impl ops::Add<Matrix<f64>> for Matrix<f64> {
    type Output = Result<Matrix<f64>>;

    fn add(self, rhs: Matrix<f64>) -> Result<Matrix<f64>> {
        if (self.rows != rhs.rows) || (self.cols != rhs.cols) {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = self.rows * self.cols;
        let output_data: Vec<f64> = (0..n_elements).into_iter()
            .map(|idx| self.data[idx] + rhs.data[idx])
            .collect::<Vec<f64>>();

        Ok(Matrix { data: output_data, rows: self.rows, cols: self.cols })
    }
}

/// Add float with Matrix.
impl ops::Add<Matrix<f32>> for f32 {
    type Output = Matrix<f32>;

    fn add(self, rhs: Matrix<f32>) -> Matrix<f32> {
        let output_data: Vec<f32> = rhs.data.iter()
            .map(|val| self + val)
            .collect::<Vec<f32>>();

        Matrix { data: output_data, rows: rhs.rows, cols: rhs.cols }
    }
}

/// Add float with Matrix.
impl ops::Add<Matrix<f64>> for f64 {
    type Output = Matrix<f64>;

    fn add(self, rhs: Matrix<f64>) -> Matrix<f64> {
        let output_data: Vec<f64> = rhs.data.iter()
            .map(|val| self + val)
            .collect::<Vec<f64>>();

        Matrix { data: output_data, rows: rhs.rows, cols: rhs.cols }
    }
}

/// Negate Matrix.
impl ops::Neg for Matrix<f32> {
    type Output = Matrix<f32>;

    fn neg(self) -> Matrix<f32> {
        let output_data: Vec<f32> = self.data.iter()
            .map(|val| -val)
            .collect::<Vec<f32>>();

        Matrix { data: output_data, rows: self.rows, cols: self.cols }
    }
}

/// Negate Matrix.
impl ops::Neg for Matrix<f64> {
    type Output = Matrix<f64>;

    fn neg(self) -> Matrix<f64> {
        let output_data: Vec<f64> = self.data.iter()
            .map(|val| -val)
            .collect::<Vec<f64>>();

        Matrix { data: output_data, rows: self.rows, cols: self.cols }
    }
}

/// Subtract float from Matrix.
impl ops::Sub<f32> for Matrix<f32> {
    type Output = Matrix<f32>;

    fn sub(self, rhs: f32) -> Matrix<f32> {
        let output_data: Vec<f32> = self.data.iter()
            .map(|val| val - rhs)
            .collect::<Vec<f32>>();

        Matrix { data: output_data, rows: self.rows, cols: self.cols }
    }
}

/// Subtract float from Matrix.
impl ops::Sub<f64> for Matrix<f64> {
    type Output = Matrix<f64>;

    fn sub(self, rhs: f64) -> Matrix<f64> {
        let output_data: Vec<f64> = self.data.iter()
            .map(|val| val - rhs)
            .collect::<Vec<f64>>();

        Matrix { data: output_data, rows: self.rows, cols: self.cols }
    }
}

/// Subtract Matrix from Matrix.
impl ops::Sub<Matrix<f32>> for Matrix<f32> {
    type Output = Result<Matrix<f32>>;

    fn sub(self, rhs: Matrix<f32>) -> Result<Matrix<f32>> {
        if (self.rows != rhs.rows) || (self.cols != rhs.cols) {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = self.rows * self.cols;
        let output_data: Vec<f32> = (0..n_elements).into_iter()
            .map(|idx| self.data[idx] - rhs.data[idx])
            .collect::<Vec<f32>>();

        Ok(Matrix { data: output_data, rows: self.rows, cols: self.cols })
    }
}

/// Subtract Matrix from Matrix.
impl ops::Sub<Matrix<f64>> for Matrix<f64> {
    type Output = Result<Matrix<f64>>;

    fn sub(self, rhs: Matrix<f64>) -> Result<Matrix<f64>> {
        if (self.rows != rhs.rows) || (self.cols != rhs.cols) {
            return Err(Jeeperr::DimensionError)
        }

        let n_elements: usize = self.rows * self.cols;
        let output_data: Vec<f64> = (0..n_elements).into_iter()
            .map(|idx| self.data[idx] - rhs.data[idx])
            .collect::<Vec<f64>>();

        Ok(Matrix { data: output_data, rows: self.rows, cols: self.cols })
    }
}

/// Subtract Matrix from float.
impl ops::Sub<Matrix<f32>> for f32 {
    type Output = Matrix<f32>;

    fn sub(self, rhs: Matrix<f32>) -> Matrix<f32> {
        let output_data: Vec<f32> = rhs.data.iter()
            .map(|val| self - val)
            .collect::<Vec<f32>>();

        Matrix { data: output_data, rows: rhs.rows, cols: rhs.cols }
    }
}

/// Subtract Matrix from float.
impl ops::Sub<Matrix<f64>> for f64 {
    type Output = Matrix<f64>;

    fn sub(self, rhs: Matrix<f64>) -> Matrix<f64> {
        let output_data: Vec<f64> = rhs.data.iter()
            .map(|val| self - val)
            .collect::<Vec<f64>>();

        Matrix { data: output_data, rows: rhs.rows, cols: rhs.cols }
    }
}

/// Multiply Matrix by float.
impl ops::Mul<f32> for Matrix<f32> {
    type Output = Matrix<f32>;

    fn mul(self, rhs: f32) -> Matrix<f32> {
        let output_data: Vec<f32> = self.data.iter()
            .map(|val| val * rhs)
            .collect::<Vec<f32>>();

        Matrix { data: output_data, rows: self.rows, cols: self.cols }
    }
}

/// Multiply Matrix by float.
impl ops::Mul<f64> for Matrix<f64> {
    type Output = Matrix<f64>;

    fn mul(self, rhs: f64) -> Matrix<f64> {
        let output_data: Vec<f64> = self.data.iter()
            .map(|val| val * rhs)
            .collect::<Vec<f64>>();

        Matrix { data: output_data, rows: self.rows, cols: self.cols }
    }
}

/// Multiply float by Matrix.
impl ops::Mul<Matrix<f32>> for f32 {
    type Output = Matrix<f32>;

    fn mul(self, rhs: Matrix<f32>) -> Matrix<f32> {
        let output_data = rhs.data.iter()
            .map(|val| self * val)
            .collect::<Vec<f32>>();

        Matrix { data: output_data, rows: rhs.rows, cols: rhs.cols }
    }
}

/// Multiply float by Matrix.
impl ops::Mul<Matrix<f64>> for f64 {
    type Output = Matrix<f64>;

    fn mul(self, rhs: Matrix<f64>) -> Matrix<f64> {
        let output_data = rhs.data.iter()
            .map(|val| self * val)
            .collect::<Vec<f64>>();

        Matrix { data: output_data, rows: rhs.rows, cols: rhs.cols }
    }
}

/// Multiply Matrix by Matrix.
impl ops::Mul<Matrix<f32>> for Matrix<f32> {
    type Output = Result<Matrix<f32>>;

    fn mul(self, rhs: Matrix<f32>) -> Result<Matrix<f32>> {
        if self.cols != rhs.rows {
            return Err(Jeeperr::DimensionError)
        }
        let new_n_elements: usize = self.rows * rhs.cols;
        let mut output_data: Vec<f32> = Vec::with_capacity(new_n_elements);

        for lhs_row in 0..self.rows {
            for rhs_col in 0..rhs.cols {
                let dot_prod: f32 = (0..self.cols).into_iter()
                    .map(|idx| self[[lhs_row, idx]] * rhs[[idx, rhs_col]])
                    .sum();

                output_data.push(dot_prod);
            }
         }

        Ok(Matrix { data: output_data, rows: self.rows, cols: rhs.cols })
    }
}

/// Multiply Matrix by Matrix.
impl ops::Mul<Matrix<f64>> for Matrix<f64> {
    type Output = Result<Matrix<f64>>;

    fn mul(self, rhs: Matrix<f64>) -> Result<Matrix<f64>> {
        if self.cols != rhs.rows {
            return Err(Jeeperr::DimensionError)
        }
        let new_n_elements: usize = self.rows * rhs.cols;
        let mut output_data: Vec<f64> = Vec::with_capacity(new_n_elements);

        for lhs_row in 0..self.rows {
            for rhs_col in 0..rhs.cols {
                let dot_prod: f64 = (0..self.cols).into_iter()
                    .map(|idx| self[[lhs_row, idx]] * rhs[[idx, rhs_col]])
                    .sum();

                output_data.push(dot_prod);
            }
         }

        Ok(Matrix { data: output_data, rows: self.rows, cols: rhs.cols })
    }
}

/// Print Matrix in a readable way.
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

pub struct RowMatIter<T: IsFloat + std::fmt::Debug + Copy + Clone> {
    row: usize,
    mat: Matrix<T>
}

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

pub struct ColMatIter<T: IsFloat + std::fmt::Debug + Copy + Clone> {
    col: usize,
    mat: Matrix<T>
}

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
