//! Calculator object allowing for GPU-accelerated matrix calculations.
//! 
//! Does not store resultant matrices in GPU memory buffers and allows for
//! temporary buffers to give more granular control over GPU memory.
//! 
//! Includes support for compilation and execution of custom kernels.

use crate::IsFloat;
use crate::Result;
use crate::Jeeperr;
use crate::linalg::Matrix;
use crate::linalg::SparseMatrix;
use crate::gpu::sparse_memory::SparseMemoryHandler;

/// Wrapper that manages storage of matrices and custom kernels and manages calculation operations.
pub struct SparseCalculator<T: IsFloat + std::fmt::Debug + Copy + Clone> {
    memory_handler: SparseMemoryHandler<T>, // Memory handler
    sparse_matrix_data: Vec<(usize, usize)>,
    dense_matrices: Vec<Matrix<T>>, // Calculator memory vector
}

impl<T: IsFloat + std::fmt::Debug + Copy + Clone> SparseCalculator<T> {
    pub fn store_sparse_matrix(&mut self, matrix: SparseMatrix<T>) -> Result<usize> {
        self.sparse_matrix_data.push((matrix.get_rows(), matrix.get_cols()));

        let output_idx: usize = self.memory_handler
            .store_sparse_matrix(matrix)?;

        if output_idx != (self.sparse_matrix_data.len() - 1) {
            return Err(Jeeperr::MemoryError)
        }

        Ok(output_idx)
    }

    /// Store matrix to calculator and gpu memory.
    pub fn store_dense_matrix(&mut self, matrix: Matrix<T>) -> Result<usize> {
        // Store matrix to calculator memory
        self.dense_matrices.push(matrix.clone());

        // Store matrix to gpu memory
        let output_idx: usize = self.memory_handler
            .store_dense_matrix(matrix)?;

        // Ensure parity between calculator and gpu memory
        if output_idx != (self.dense_matrices.len() - 1) {
            return Err(Jeeperr::MemoryError)
        }

        Ok(output_idx)
    }
}

impl SparseCalculator<f64> {
    /// Initializes Calculator struct.
    pub fn init() -> Result<SparseCalculator<f64>> {
        // Initialize vector of kernel names
        let program_vec: Vec<&str> = super::SPARSE_LIST_DOUBLE.to_vec();
        
        // Create memory handler using program source and kernel names
        let memory_handler: SparseMemoryHandler<f64> = SparseMemoryHandler::<f64>::init(super::SPARSE_SOURCE_DOUBLE, program_vec)?;

        // Create empty memory vector
        let sparse_mat_vector: Vec<(usize, usize)> = Vec::with_capacity(super::INIT_MEMORY_CAPACITY);
        let dense_mat_vector: Vec<Matrix<f64>> = Vec::with_capacity(super::INIT_MEMORY_CAPACITY);

        // Create and return new quick calculator
        let output: SparseCalculator<f64> = SparseCalculator {
            memory_handler: memory_handler,
            sparse_matrix_data: sparse_mat_vector,
            dense_matrices: dense_mat_vector
        };
        Ok(output)
    }

    /// Multiply previously stored Matrix and previously stored Matrix.
    pub fn mat_mul(&mut self, left_sparse_idx: usize, right_dense_matrix: &Matrix<f64>) -> Result<Matrix<f64>> {
        let (rows, cols) = self.sparse_matrix_data[left_sparse_idx];

        if cols != right_dense_matrix.get_rows() {
            return Err(Jeeperr::DimensionError)
        }

        let cols: usize = right_dense_matrix.get_cols();

        let output: Matrix<f64> = self.memory_handler.execute_once_and_read(
            0,
            None,
            Some(vec![cols as i32]),
            vec![left_sparse_idx],
            None,
            Some(vec![&right_dense_matrix]),
            rows,
            cols,
            vec![rows, cols]
        )?;

        Ok(output)
    }
}