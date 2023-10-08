//! Calculator object allowing for GPU-accelerated matrix calculations
//! 
//! Stores input and resultant matrices in GPU memory buffers for repeated use
//! 
//! Includes support for compilation and execution of custom kernels

use crate::Result;
use crate::Jeeperr;
use crate::linalg::Matrix;
use crate::gpu::memory::MemoryHandler;

type ResultFunction = Box<dyn Fn(
    usize,
    &mut MemoryHandler,
    Option<Vec<f32>>,
    Option<Vec<i32>>,
    Vec<usize>,
    usize,
    usize,
    Vec<usize>
) -> Result<(Matrix, usize)>>;

/// Shortcut type definition for closure defining output parameters for custom kernel
pub type ParameterFunction = Box<dyn Fn(Vec<&Matrix>) -> Result<(usize, usize, Vec<usize>)>>;

/// Wrapper that manages storage of matrices and custom kernels and manages calculation operations
pub struct Calculator {
    memory_handler: MemoryHandler, // Memory handler
    matrices: Vec<Matrix>, // Calculator memory vector
    customs: Vec<ResultFunction>,
    params_customs: Vec<ParameterFunction>,
    custom_idcs: Vec<usize>
}

impl Calculator {
    /// Initializes Calculator struct
    pub fn init() -> Result<Calculator> {
        // Initialize vector of kernel names
        let program_vec: Vec<&str> = super::PROGRAM_LIST.to_vec();
        
        // Create memory handler using program source and kernel names
        let memory_handler: MemoryHandler = MemoryHandler::new(super::PROGRAM_SOURCE, program_vec)?;

        // Create empty memory vector
        let mat_vector: Vec<Matrix> = Vec::with_capacity(super::INIT_MEMORY_CAPACITY);

        let customs_vector: Vec<ResultFunction> = Vec::default();
        let params_customs_vector: Vec<ParameterFunction> = Vec::default();
        let custom_idcs_vector: Vec<usize> = Vec::default();

        // Create and return new memory calculator
        let output: Calculator = Calculator {
            memory_handler: memory_handler,
            matrices: mat_vector,
            customs: customs_vector,
            params_customs: params_customs_vector,
            custom_idcs: custom_idcs_vector
        };
        Ok(output)
    }

    /// Store matrix to calculator and gpu memory
    pub fn store_matrix(&mut self, matrix: Matrix) -> Result<usize> {
        // Store matrix to calculator memory
        self.matrices.push(matrix.clone());

        // Store matrix to gpu memory
        let output_idx: usize = self.memory_handler
            .store_matrix(matrix)?;

        // Ensure parity between calculator and gpu memory
        if output_idx != (self.matrices.len() - 1) {
            return Err(Jeeperr::MemoryError)
        }

        Ok(output_idx)
    }

    /// Multiply Matrix and Matrix
    pub fn mat_mul(&mut self, left_idx: usize, right_idx: usize) -> Result<(Matrix, usize)> {
        let left: &Matrix = &self.matrices[left_idx];
        let right: &Matrix = &self.matrices[right_idx];

        if left.get_cols() != right.get_rows() {
            return Err(Jeeperr::DimensionError)
        }

        let rows: usize = left.get_rows();
        let interm: usize = left.get_cols();
        let cols: usize = right.get_cols();

        let (output, output_idx) = self.memory_handler.execute_and_read(
            0,
            None,
            Some(vec![cols as i32, interm as i32]),
            vec![left_idx, right_idx],
            rows,
            cols,
            vec![rows, cols]
        )?;

        self.matrices.push(output.clone());

        Ok((output, output_idx))
    }

    /// Compile and store a custom kernel and build/store a closure to execute said kernel
    pub unsafe fn load_custom_fn(
        &mut self,
        program_source: &str,
        kernel_name: &str,
        parameter_fn: ParameterFunction
    ) -> Result<usize>
    {
        let new_kernel_index: usize = self.memory_handler.new_kernel(program_source, kernel_name)?;
        self.custom_idcs.push(new_kernel_index);

        self.params_customs.push(parameter_fn);

        let new_custom: ResultFunction = Box::new(|
                index: usize,
                host: &mut MemoryHandler,
                input_floats: Option<Vec<f32>>,
                input_ints: Option<Vec<i32>>,
                input_mat_idcs: Vec<usize>,
                output_rows: usize,
                output_cols: usize,
                work_sizes: Vec<usize>
            | -> Result<(Matrix, usize)>
            {
                let (output, output_idx) = host.execute_and_read(
                    index,
                    input_floats,
                    input_ints,
                    input_mat_idcs,
                    output_rows,
                    output_cols,
                    work_sizes
                )?;

                Ok((output, output_idx))
            }
        );

        self.customs.push(new_custom);

        return Ok(self.customs.len() - 1);
    }

    /// Execute custom kernel via pre-generated closure
    pub unsafe fn exec_custom_fn(
        &mut self,
        custom_index: usize,
        input_floats: Option<Vec<f32>>,
        input_ints: Option<Vec<i32>>,
        input_mat_idcs: Vec<usize>
    ) -> Result<(Matrix, usize)>
    {
        let input_matrices: Vec<&Matrix> = input_mat_idcs
            .clone()
            .into_iter()
            .map(| idx | &self.matrices[idx])
            .collect::<Vec<_>>();

        let (output_rows, output_cols, work_sizes) = self.params_customs[custom_index](input_matrices)?;

        let (output_matrix, output_idx) = self.customs[custom_index](
            self.custom_idcs[custom_index],
            &mut self.memory_handler,
            input_floats,
            input_ints,
            input_mat_idcs,
            output_rows,
            output_cols,
            work_sizes
        )?;

        self.matrices.push(output_matrix.clone());

        if output_idx != (self.matrices.len() - 1) {
            return Err(Jeeperr::MemoryError);
        }

        Ok((output_matrix, output_idx))
    }
}