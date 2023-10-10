//! Matrix object and implementations

use std::ops;

use crate::Result;
use crate::Jeeperr;

/// Float matrix struct
#[derive(Debug, Clone)]
pub struct Matrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize
}

impl Matrix {
    /// Create new matrix (includes error checking)
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Result<Matrix> {
        let vec_len: usize = data.len();
        let comp_len: usize = rows * cols;

        if vec_len == comp_len {
            let output: Matrix = Matrix { data, rows, cols };

            return Ok(output)
        }
        else {
            return Err(Jeeperr::DimensionError)
        }
    }
    
    /// Create new matrix of zeros
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        let data: Vec<f32> = vec![0.0; rows * cols];

        return Matrix { data, rows, cols }
    }

    /// Create new matrix of ones
    pub fn ones(rows: usize, cols: usize) -> Matrix {
        let data: Vec<f32> = vec![1.0; rows * cols];

        return Matrix { data, rows, cols }
    }

    pub fn identity(dim_n: usize) -> Matrix {
        let mut data: Vec<f32> = vec![0.0; dim_n * dim_n];
        
        for diagonal_idx in 0..dim_n {
            data[diagonal_idx * dim_n + diagonal_idx] = 1.0;
        }

        return Matrix { data, rows: dim_n, cols: dim_n };
    }

    /// Get number of rows in matrix
    pub fn get_rows(&self) -> usize {
        return self.rows
    }

    /// Get number of cols in matrix
    pub fn get_cols(&self) -> usize {
        return self.cols
    }

    /// Get matrix data in slice form
    pub fn get_data(&self) -> &[f32] {
        return &self.data
    }

    /// Get indexed row of matrix in matrix form
    pub fn row_matrix(&self, row_idx: usize) -> Result<Matrix> {
        if row_idx >= self.rows {
            return Err(Jeeperr::IndexError)
        }

        let lower_data_idx: usize = row_idx * self.cols;
        let upper_data_idx: usize = (row_idx + 1) * self.cols;

        let output_row_data: Vec<f32> = self.data[lower_data_idx..upper_data_idx].to_vec();

        return Ok(Matrix { data: output_row_data, rows: 1, cols: self.cols })
    }

    /// Get indexed row of matrix in vector form
    pub fn row_vec(&self, row_idx: usize) -> Result<Vec<f32>> {
        if row_idx >= self.rows {
            return Err(Jeeperr::IndexError)
        }

        let lower_data_idx: usize = row_idx * self.cols;
        let upper_data_idx: usize = (row_idx + 1) * self.cols;

        let output_row: Vec<f32> = self.data[lower_data_idx..upper_data_idx].to_vec();

        return Ok(output_row)
    }

    /// Get indexed col of matrix in matrix form
    pub fn col_matrix(&self, col_idx: usize) -> Result<Matrix> {
        if col_idx >= self.cols {
            return Err(Jeeperr::IndexError)
        }

        let lower_data_idx: usize = col_idx;
        let upper_data_idx: usize = (self.rows - 1) * self.cols + col_idx;

        let output_col_data: Vec<f32> = (lower_data_idx..=upper_data_idx).step_by(self.cols)
            .map(|idx| self.data[idx])
            .collect::<Vec<f32>>();

        return Ok(Matrix { data: output_col_data, rows: self.rows, cols: 1 })
    }

    /// Get indexed col of matrix in vector form
    pub fn col_vec(&self, col_idx: usize) -> Result<Vec<f32>> {
        if col_idx >= self.cols {
            return Err(Jeeperr::IndexError)
        }

        let lower_data_idx: usize = col_idx;
        let upper_data_idx: usize = self.rows * self.cols;

        let output_col: Vec<f32> = (lower_data_idx..upper_data_idx).step_by(self.cols)
            .map(|idx| self.data[idx])
            .collect::<Vec<f32>>();

        return Ok(output_col)
    }

    /// Index matrix using usize slices, outputing matrix of the index intersections
    pub fn slice_index(&self, row_idcs: &[usize], col_idcs: &[usize]) -> Matrix {
        let output_rows: usize = row_idcs.len();
        let output_cols: usize = col_idcs.len();
        let mut output_data: Vec<f32> = Vec::with_capacity(output_rows * output_cols);
        for row_idx in row_idcs {
            for col_idx in col_idcs {
                output_data.push(self.data[row_idx * self.cols + col_idx]);
            }
        }

        return Matrix { data: output_data, rows: output_rows, cols: output_cols }
    }

    /// Matrix transpose
    pub fn transpose(self) -> Matrix {
        let mut transpose_data: Vec<f32> = Vec::with_capacity(self.rows * self.cols);

        for transpose_row in 0..self.cols {
            for transpose_col in 0..self.rows {
                transpose_data.push(self[[transpose_col, transpose_row]]);
            }
        }

        Matrix { data: transpose_data, rows: self.cols, cols: self.rows }
    }

    /// Elementwise multiplication between two matrices
    pub fn elementwise(self, rhs: Matrix) -> Result<Matrix> {
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

/// Index using the form matrix[[row, col]]
impl ops::Index<[usize; 2]> for Matrix {
    type Output = f32;

    fn index(&self, idx: [usize; 2]) -> &f32 {
        return &self.data[idx[0] * self.cols + idx[1]]
    }
}

/// Add matrix with float
impl ops::Add<f32> for Matrix {
    type Output = Matrix;

    fn add(self, rhs: f32) -> Matrix {
        let output_data: Vec<f32> = self.data.iter()
            .map(|val| val + rhs)
            .collect::<Vec<f32>>();

        Matrix { data: output_data, rows: self.rows, cols: self.cols }
    }
}

/// Add Matrix with matrix
impl ops::Add<Matrix> for Matrix {
    type Output = Result<Matrix>;

    fn add(self, rhs: Matrix) -> Result<Matrix> {
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

/// Add float with matrix
impl ops::Add<Matrix> for f32 {
    type Output = Matrix;

    fn add(self, rhs: Matrix) -> Matrix {
        let output_data: Vec<f32> = rhs.data.iter()
            .map(|val| self + val)
            .collect::<Vec<f32>>();

        Matrix { data: output_data, rows: rhs.rows, cols: rhs.cols }
    }
}

/// Negate matrix
impl ops::Neg for Matrix {
    type Output = Matrix;

    fn neg(self) -> Matrix {
        let output_data: Vec<f32> = self.data.iter()
            .map(|val| -val)
            .collect::<Vec<f32>>();

        Matrix { data: output_data, rows: self.rows, cols: self.cols }
    }
}

/// Subtract float from matrix
impl ops::Sub<f32> for Matrix {
    type Output = Matrix;

    fn sub(self, rhs: f32) -> Matrix {
        let output_data: Vec<f32> = self.data.iter()
            .map(|val| val - rhs)
            .collect::<Vec<f32>>();

        Matrix { data: output_data, rows: self.rows, cols: self.cols }
    }
}

/// Subtract matrix from matrix
impl ops::Sub<Matrix> for Matrix {
    type Output = Result<Matrix>;

    fn sub(self, rhs: Matrix) -> Result<Matrix> {
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

/// Subtract matrix from float
impl ops::Sub<Matrix> for f32 {
    type Output = Matrix;

    fn sub(self, rhs: Matrix) -> Matrix {
        let output_data: Vec<f32> = rhs.data.iter()
            .map(|val| self - val)
            .collect::<Vec<f32>>();

        Matrix { data: output_data, rows: rhs.rows, cols: rhs.cols }
    }
}

/// Multiply matrix by float
impl ops::Mul<f32> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: f32) -> Matrix {
        let output_data: Vec<f32> = self.data.iter()
            .map(|val| val * rhs)
            .collect::<Vec<f32>>();

        Matrix { data: output_data, rows: self.rows, cols: self.cols }
    }
}

/// Multiply float by matrix
impl ops::Mul<Matrix> for f32 {
    type Output = Matrix;

    fn mul(self, rhs: Matrix) -> Matrix {
        let output_data = rhs.data.iter()
            .map(|val| self * val)
            .collect::<Vec<f32>>();

        Matrix { data: output_data, rows: rhs.rows, cols: rhs.cols }
    }
}

/// Multiply matrix by matrix
impl ops::Mul<Matrix> for Matrix {
    type Output = Result<Matrix>;

    fn mul(self, rhs: Matrix) -> Result<Matrix> {
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

/// Print matrix in a readable way
impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "[{:?}", self.row_vec(0).unwrap())?;
        for row_idx in 1..self.rows {
            write!(f, " {:?}", self.row_vec(row_idx).unwrap())?;
            if row_idx < self.rows - 1 {
                write!(f, "\n")?;
            }
        }
        writeln!(f, "]")?;

        Ok(())
    }
}