use std::ops;
use std::collections::HashMap;

use crate::IsFloat;
use crate::Result;
use crate::Jeeperr;

use crate::linalg::Matrix;

#[derive(Debug, Clone)]
pub struct SparseMatrix<T: IsFloat + std::fmt::Debug + Copy + Clone> {
    map: HashMap<usize, HashMap<usize, T>>,
    rows: usize,
    cols: usize
}

impl<T: IsFloat + std::fmt::Debug + Copy + Clone> SparseMatrix<T> {
    pub fn empty(rows: usize, cols: usize) -> SparseMatrix<T> {
        let map: HashMap<usize, HashMap<usize, T>> = HashMap::new();

        return SparseMatrix { map, rows, cols };
    }

    pub fn get_rows(&self) -> usize {
        return self.rows;
    }

    pub fn get_cols(&self) -> usize {
        return self.cols;
    }

    pub fn reassign_at(&mut self, index: [usize; 2], value: T) -> Result<()> {
        if index[0] < self.rows && index[1] < self.cols {
            match self.map.get_mut(&index[0]) {
                Some(row_map) => { row_map.insert(index[1], value); },
                None => { self.map.insert(index[0], HashMap::from([(index[1], value)])); }
            };

            return Ok(());
        }
        else {
            return Err(Jeeperr::IndexError)
        }
    }
}

impl SparseMatrix<f32> {
    pub fn from_dense(dense: Matrix<f32>) -> SparseMatrix<f32> {
        let mut map: HashMap<usize, HashMap<usize, f32>> = HashMap::new();

        let rows: usize = dense.get_rows();
        let cols: usize = dense.get_cols();

        for i in 0..rows {
            for j in 0..cols {
                if dense[[i, j]] != 0.0 {
                    if map.contains_key(&i) {
                        map.get_mut(&i).unwrap().insert(j, dense[[i, j]]);
                    }
                    else {
                        map.insert(i, HashMap::from([(j, dense[[i, j]])]));
                    }
                }
            }
        }

        return SparseMatrix { map, rows, cols }
    }

    pub fn increment_at(&mut self, index: [usize; 2], value: f32) -> Result<f32> {
        if index[0] < self.rows && index[1] < self.cols {
            if self.map.contains_key(&index[0]) {
                let row_map: &mut HashMap<usize, f32> = self.map.get_mut(&index[0]).unwrap();
                match &mut row_map.get(&index[1]) {
                    Some(&current_value) => {
                        row_map.insert(index[1], current_value + value);
                        return Ok(current_value + value);
                    },
                    None => {
                        row_map.insert(index[1], value);
                        return Ok(value)
                    }
                }
            }
            else {
                self.map.insert(index[0], HashMap::from([(index[1], value)]));
                return Ok(value)
            }
        }
        else {
            return Err(Jeeperr::IndexError)
        }
    }

    pub fn multiply_at(&mut self, index: [usize; 2], value: f32) -> Result<f32> {
        if index[0] < self.rows && index[1] < self.cols {
            if self.map.contains_key(&index[0]) {
                let row_map: &mut HashMap<usize, f32> = self.map.get_mut(&index[0]).unwrap();
                match &mut row_map.get(&index[1]) {
                    Some(&current_value) => {
                        row_map.insert(index[1], current_value * value);
                        return Ok(current_value * value);
                    },
                    None => {
                        row_map.insert(index[1], value);
                        return Ok(value)
                    }
                }
            }
            else {
                self.map.insert(index[0], HashMap::from([(index[1], value)]));
                return Ok(value)
            }
        }
        else {
            return Err(Jeeperr::IndexError)
        }
    }

    pub fn row_matrix(&self, row_idx: usize) -> Result<Matrix<f32>> {
        if row_idx < self.rows {
            if self.map.contains_key(&row_idx) {
                let mut output_row: Matrix<f32> = Matrix::<f32>::zeros(1, self.cols);

                self.map.get(&row_idx)
                    .unwrap()
                    .iter()
                    .for_each(|(&key, &val)| output_row[[0, key]] = val);

                return Ok(output_row)
            }
            else {
                return Ok(Matrix::<f32>::zeros(1, self.cols));
            }
        }
        else {
            return Err(Jeeperr::IndexError)
        }
    }
}

impl SparseMatrix<f64> {
    pub fn from_dense(dense: Matrix<f64>) -> SparseMatrix<f64> {
        let mut map: HashMap<usize, HashMap<usize, f64>> = HashMap::new();

        let rows: usize = dense.get_rows();
        let cols: usize = dense.get_cols();

        for i in 0..rows {
            for j in 0..cols {
                if dense[[i, j]] != 0.0 {
                    if map.contains_key(&i) {
                        map.get_mut(&i).unwrap().insert(j, dense[[i, j]]);
                    }
                    else {
                        map.insert(i, HashMap::from([(j, dense[[i, j]])]));
                    }
                }
            }
        }

        return SparseMatrix { map, rows, cols }
    }

    pub fn increment_at(&mut self, index: [usize; 2], value: f64) -> Result<f64> {
        if index[0] < self.rows && index[1] < self.cols {
            if self.map.contains_key(&index[0]) {
                let row_map: &mut HashMap<usize, f64> = self.map.get_mut(&index[0]).unwrap();
                match &mut row_map.get(&index[1]) {
                    Some(&current_value) => {
                        row_map.insert(index[1], current_value + value);
                        return Ok(current_value + value);
                    },
                    None => {
                        row_map.insert(index[1], value);
                        return Ok(value)
                    }
                }
            }
            else {
                self.map.insert(index[0], HashMap::from([(index[1], value)]));
                return Ok(value)
            }
        }
        else {
            return Err(Jeeperr::IndexError)
        }
    }

    pub fn multiply_at(&mut self, index: [usize; 2], value: f64) -> Result<f64> {
        if index[0] < self.rows && index[1] < self.cols {
            if self.map.contains_key(&index[0]) {
                let row_map: &mut HashMap<usize, f64> = self.map.get_mut(&index[0]).unwrap();
                match &mut row_map.get(&index[1]) {
                    Some(&current_value) => {
                        row_map.insert(index[1], current_value * value);
                        return Ok(current_value * value);
                    },
                    None => {
                        row_map.insert(index[1], value);
                        return Ok(value)
                    }
                }
            }
            else {
                self.map.insert(index[0], HashMap::from([(index[1], value)]));
                return Ok(value)
            }
        }
        else {
            return Err(Jeeperr::IndexError)
        }
    }

    pub fn row_matrix(&self, row_idx: usize) -> Result<Matrix<f64>> {
        if row_idx < self.rows {
            if self.map.contains_key(&row_idx) {
                let mut output_row: Matrix<f64> = Matrix::<f64>::zeros(1, self.cols);

                self.map.get(&row_idx)
                    .unwrap()
                    .iter()
                    .for_each(|(&key, &val)| output_row[[0, key]] = val);

                return Ok(output_row)
            }
            else {
                return Ok(Matrix::<f64>::zeros(1, self.cols));
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
        return match self.map.get(&index[0]) {
            Some(row_map) => match row_map.get(&index[1]) {
                Some(current_value) => current_value,
                None => &0.0
            },
            None => &0.0
        }
    }
}

impl ops::Index<[usize; 2]> for SparseMatrix<f64> {
    type Output = f64;

    fn index(&self, index: [usize; 2]) -> &f64 {
        return match self.map.get(&index[0]) {
            Some(row_map) => match row_map.get(&index[1]) {
                Some(current_value) => current_value,
                None => &0.0
            },
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

        for row in self.map.keys() {
            for col in 0..rhs.get_cols() {
                let row_map: &HashMap<usize, f32> = self.map.get(row).unwrap();
                output_data[row * rhs.get_cols() + col] = row_map.iter()
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

        for row in self.map.keys() {
            for col in 0..rhs.get_cols() {
                let row_map: &HashMap<usize, f64> = self.map.get(row).unwrap();
                output_data[row * rhs.get_cols() + col] = row_map.iter()
                    .map(|(&lhs_col, lhs_val)| lhs_val * rhs[[lhs_col, col]])
                    .sum();
            }
        }

        return Ok(Matrix::new(output_data, self.rows, rhs.get_cols())?)
    }
}