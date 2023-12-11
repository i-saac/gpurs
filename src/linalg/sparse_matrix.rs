use std::ops;
use std::collections::HashMap;

use crate::IsFloat;
use crate::Result;
use crate::Jeeperr;

use crate::linalg::Matrix;

#[derive(Debug, Clone)]
pub struct SparseMatrix<T: IsFloat + std::fmt::Debug + Copy + Clone> {
    data: Vec<HashMap<usize, T>>,
    rows: usize,
    cols: usize
}

impl<T: IsFloat + std::fmt::Debug + Copy + Clone> SparseMatrix<T> {
    pub fn empty(rows: usize, cols: usize) -> SparseMatrix<T> {
        let data: Vec<HashMap<usize, T>> = vec![HashMap::new(); rows];

        return SparseMatrix { data, rows, cols };
    }

    pub fn get_rows(&self) -> usize {
        return self.rows;
    }

    pub fn get_cols(&self) -> usize {
        return self.cols;
    }

    pub fn get_data(&self) -> &Vec<HashMap<usize, T>> {
        return &self.data;
    }

    pub fn reassign_at(&mut self, index: [usize; 2], value: T) -> Result<()> {
        if index[0] < self.rows && index[1] < self.cols {
            self.data[index[0]].insert(index[1], value);

            return Ok(());
        }
        else {
            return Err(Jeeperr::IndexError)
        }
    }
}

impl SparseMatrix<f32> {
    pub fn from_dense(dense: Matrix<f32>) -> SparseMatrix<f32> {
        let rows: usize = dense.get_rows();
        let cols: usize = dense.get_cols();

        let mut data: Vec<HashMap<usize, f32>> = Vec::with_capacity(dense.get_rows());
        for row in dense.all_rows() {
            let mut row_map: HashMap<usize, f32> = HashMap::new();

            row.get_data()
                .into_iter()
                .enumerate()
                .filter(|(_, &item)| item != 0.0)
                .for_each(|(col_idx, &item)| { row_map.insert(col_idx, item); });

            data.push(row_map);
        }

        return SparseMatrix { data, rows, cols }
    }

    pub fn increment_at(&mut self, index: [usize; 2], value: f32) -> Result<f32> {
        if index[0] < self.rows && index[1] < self.cols {
            match self.data[index[0]].get(&index[1]) {
                Some(&current_value) => {
                    self.data[index[0]].insert(index[1], current_value + value);
                    return Ok(current_value + value);
                },
                None => {
                    self.data[index[0]].insert(index[1], value);
                    return Ok(value)
                }
            }
        }
        else {
            return Err(Jeeperr::IndexError)
        }
    }

    pub fn multiply_at(&mut self, index: [usize; 2], value: f32) -> Result<f32> {
        if index[0] < self.rows && index[1] < self.cols {
            match self.data[index[0]].get(&index[1]) {
                Some(&current_value) => {
                    self.data[index[0]].insert(index[1], current_value * value);
                    return Ok(current_value * value);
                },
                None => {
                    self.data[index[0]].insert(index[1], value);
                    return Ok(value)
                }
            }
        }
        else {
            return Err(Jeeperr::IndexError)
        }
    }
}

impl SparseMatrix<f64> {
    pub fn from_dense(dense: Matrix<f64>) -> SparseMatrix<f64> {
        let rows: usize = dense.get_rows();
        let cols: usize = dense.get_cols();

        let mut data: Vec<HashMap<usize, f64>> = Vec::with_capacity(dense.get_rows());
        for row in dense.all_rows() {
            let mut row_map: HashMap<usize, f64> = HashMap::new();

            row.get_data()
                .into_iter()
                .enumerate()
                .filter(|(_, &item)| item != 0.0)
                .for_each(|(col_idx, &item)| { row_map.insert(col_idx, item); });

            data.push(row_map);
        }

        return SparseMatrix { data, rows, cols }
    }

    pub fn increment_at(&mut self, index: [usize; 2], value: f64) -> Result<f64> {
        if index[0] < self.rows && index[1] < self.cols {
            match self.data[index[0]].get(&index[1]) {
                Some(&current_value) => {
                    self.data[index[0]].insert(index[1], current_value + value);
                    return Ok(current_value + value);
                },
                None => {
                    self.data[index[0]].insert(index[1], value);
                    return Ok(value)
                }
            }
        }
        else {
            return Err(Jeeperr::IndexError)
        }
    }

    pub fn multiply_at(&mut self, index: [usize; 2], value: f64) -> Result<f64> {
        if index[0] < self.rows && index[1] < self.cols {
            match self.data[index[0]].get(&index[1]) {
                Some(&current_value) => {
                    self.data[index[0]].insert(index[1], current_value * value);
                    return Ok(current_value * value);
                },
                None => {
                    self.data[index[0]].insert(index[1], value);
                    return Ok(value)
                }
            }
        }
        else {
            return Err(Jeeperr::IndexError)
        }
    }
}

impl ops::Index<[usize; 2]> for SparseMatrix<f32> {
    type Output = f32;

    fn index(&self, index: [usize; 2]) -> &f32 {
        return match self.data[index[0]].get(&index[1]) {
            Some(current_value) => current_value,
            None => &0.0
        }
    }
}

impl ops::Index<[usize; 2]> for SparseMatrix<f64> {
    type Output = f64;

    fn index(&self, index: [usize; 2]) -> &f64 {
        return match self.data[index[0]].get(&index[1]) {
            Some(current_value) => current_value,
            None => &0.0
        }
    }
}

impl ops::Mul<Matrix<f32>> for SparseMatrix<f32> {
    type Output = Result<Matrix<f32>>;

    fn mul(self, rhs: Matrix<f32>) -> Result<Matrix<f32>> {
        if self.cols != rhs.get_rows() {
            return Err(Jeeperr::DimensionError)
        }

        let new_n_elements: usize = self.rows * rhs.get_cols();
        let mut output_data: Vec<f32> = vec![0.0; new_n_elements];

        for (row_idx, row_map) in self.data.iter().enumerate() {
            for col in 0..rhs.get_cols() {
                output_data[row_idx * rhs.get_cols() + col] = row_map.iter()
                    .map(|(&lhs_col, lhs_val)| lhs_val * rhs[[lhs_col, col]])
                    .sum();
            }
        }

        return Ok(Matrix::new(output_data, self.rows, rhs.get_cols())?)
    }
}

impl ops::Mul<Matrix<f64>> for SparseMatrix<f64> {
    type Output = Result<Matrix<f64>>;

    fn mul(self, rhs: Matrix<f64>) -> Result<Matrix<f64>> {
        if self.cols != rhs.get_rows() {
            return Err(Jeeperr::DimensionError)
        }

        let new_n_elements: usize = self.rows * rhs.get_cols();
        let mut output_data: Vec<f64> = vec![0.0; new_n_elements];

        for (row_idx, row_map) in self.data.iter().enumerate() {
            for col in 0..rhs.get_cols() {
                output_data[row_idx * rhs.get_cols() + col] = row_map.iter()
                    .map(|(&lhs_col, lhs_val)| lhs_val * rhs[[lhs_col, col]])
                    .sum();
            }
        }

        return Ok(Matrix::new(output_data, self.rows, rhs.get_cols())?)
    }
}