//! Calculator object allowing for GPU-accelerated matrix calculations
//! 
//! Does not store resultant matrices in GPU memory buffers and allows for
//! temporary buffers to give more granular control over GPU memory
//! 
//! Includes support for compilation and execution of custom kernels

use crate::IsFloat;
use crate::Result;
use crate::Jeeperr;
use crate::linalg::Matrix;
use crate::gpu::memory::MemoryHandler;

type ResultFunction<T> = Box<dyn Fn(
    usize,
    &mut MemoryHandler<T>,
    Option<Vec<T>>,
    Option<Vec<i32>>,
    Option<Vec<usize>>,
    Option<Vec<&Matrix<T>>>,
    usize,
    usize,
    Vec<usize>
) -> Result<Matrix<T>>>;

/// Shortcut type definition for closure defining output parameters for custom kernel
/// Inputs is a list of matrices, the output is a Result containing a tuple of the output rows, output cols, and work sizes
pub type QuickParameterFunction<T> = Box<dyn Fn(Option<Vec<&Matrix<T>>>, Option<Vec<&Matrix<T>>>) -> Result<(usize, usize, Vec<usize>)>>;

/// Wrapper that manages storage of matrices and custom kernels and manages calculation operations
pub struct QuickCalculator<T: IsFloat + std::fmt::Debug + Copy + Clone> {
    memory_handler: MemoryHandler<T>, // Memory handler
    matrices: Vec<Matrix<T>>, // Calculator memory vector
    customs: Vec<ResultFunction<T>>,
    params_customs: Vec<QuickParameterFunction<T>>,
    custom_idcs: Vec<usize>
}

impl<T: IsFloat + std::fmt::Debug + Copy + Clone> QuickCalculator<T> {
    /// Store matrix to calculator and gpu memory
    pub fn store_matrix(&mut self, matrix: Matrix<T>) -> Result<usize> {
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
}

impl QuickCalculator<f32> {
    /// Initializes Calculator struct
    pub fn init() -> Result<QuickCalculator<f32>> {
        // Initialize vector of kernel names
        let program_vec: Vec<&str> = super::PROGRAM_LIST_FLOAT.to_vec();
        
        // Create memory handler using program source and kernel names
        let memory_handler: MemoryHandler<f32> = MemoryHandler::<f32>::init(super::PROGRAM_SOURCE_FLOAT, program_vec)?;

        // Create empty memory vector
        let mat_vector: Vec<Matrix<f32>> = Vec::with_capacity(super::INIT_MEMORY_CAPACITY);

        let customs_vector: Vec<ResultFunction<f32>> = Vec::default();
        let params_customs_vector: Vec<QuickParameterFunction<f32>> = Vec::default();
        let custom_idcs_vector: Vec<usize> = Vec::default();

        // Create and return new memory calculator
        let output: QuickCalculator<f32> = QuickCalculator {
            memory_handler: memory_handler,
            matrices: mat_vector,
            customs: customs_vector,
            params_customs: params_customs_vector,
            custom_idcs: custom_idcs_vector
        };
        Ok(output)
    }

    /// Multiply previously stored Matrix and previously stored Matrix
    pub fn mat_mul(&mut self, left_idx: usize, right_idx: usize) -> Result<Matrix<f32>> {
        let left: &Matrix<f32> = &self.matrices[left_idx];
        let right: &Matrix<f32> = &self.matrices[right_idx];

        if left.get_cols() != right.get_rows() {
            return Err(Jeeperr::DimensionError)
        }

        let rows: usize = left.get_rows();
        let interm: usize = left.get_cols();
        let cols: usize = right.get_cols();

        let output: Matrix<f32> = self.memory_handler.execute_once_and_read(
            0,
            None,
            Some(vec![cols as i32, interm as i32]),
            Some(vec![left_idx, right_idx]),
            None,
            rows,
            cols,
            vec![rows, cols]
        )?;

        Ok(output)
    }

    pub fn halfquick_mat_mul(&mut self, left_idx: usize, right: &Matrix<f32>) -> Result<Matrix<f32>> {
        let left: &Matrix<f32> = &self.matrices[left_idx];

        if left.get_cols() != right.get_rows() {
            return Err(Jeeperr::DimensionError)
        }

        let rows: usize = left.get_rows();
        let interm: usize = left.get_cols();
        let cols: usize = right.get_cols();

        let output: Matrix<f32> = self.memory_handler.execute_once_and_read(
            0,
            None,
            Some(vec![cols as i32, interm as i32]),
            Some(vec![left_idx]),
            Some(vec![right]),
            rows,
            cols,
            vec![rows, cols]
        )?;

        Ok(output)
    }

    /// Multiply non-stored Matrix and non-stored Matrix
    pub fn quick_mat_mul(&mut self, left: &Matrix<f32>, right: &Matrix<f32>) -> Result<Matrix<f32>> {
        if left.get_cols() != right.get_rows() {
            return Err(Jeeperr::DimensionError)
        }

        let rows: usize = left.get_rows();
        let interm: usize = left.get_cols();
        let cols: usize = right.get_cols();

        let output: Matrix<f32> = self.memory_handler.execute_once_and_read(
            0,
            None,
            Some(vec![cols as i32, interm as i32]),
            None,
            Some(vec![left, right]),
            rows,
            cols,
            vec![rows, cols]
        )?;

        Ok(output)
    }

    /// Compile and store a custom kernel and build/store a closure to execute said kernel
    pub unsafe fn load_custom_fn(
        &mut self,
        program_source: &str,
        kernel_name: &str,
        parameter_fn: QuickParameterFunction<f32>
    ) -> Result<usize>
    {
        let new_kernel_index: usize = self.memory_handler.new_kernel(program_source, kernel_name)?;
        self.custom_idcs.push(new_kernel_index);

        self.params_customs.push(parameter_fn);

        let new_custom: ResultFunction<f32> = Box::new(|
                index: usize,
                host: &mut MemoryHandler<f32>,
                input_floats: Option<Vec<f32>>,
                input_ints: Option<Vec<i32>>,
                input_mat_idcs: Option<Vec<usize>>,
                input_temp_mats: Option<Vec<&Matrix<f32>>>,
                output_rows: usize,
                output_cols: usize,
                work_sizes: Vec<usize>
            | -> Result<Matrix<f32>>
            {
                let output = host.execute_once_and_read(
                    index,
                    input_floats,
                    input_ints,
                    input_mat_idcs,
                    input_temp_mats,
                    output_rows,
                    output_cols,
                    work_sizes
                )?;

                Ok(output)
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
        input_mat_idcs: Option<Vec<usize>>,
        input_temp_mats: Option<Vec<&Matrix<f32>>>
    ) -> Result<Matrix<f32>>
    {
        let input_matrices: Option<Vec<&Matrix<f32>>>;
        if input_mat_idcs.is_some() {
            input_matrices = Some(input_mat_idcs.clone()
                .unwrap()
                .clone()
                .iter()
                .map(| &idx | &self.matrices[idx])
                .collect::<Vec<&Matrix<f32>>>());
        }
        else {
            input_matrices = None;
        }

        let (output_rows, output_cols, work_sizes) = self.params_customs[custom_index](input_matrices.clone(), input_temp_mats.clone())?;

        let output_matrix = self.customs[custom_index](
            self.custom_idcs[custom_index],
            &mut self.memory_handler,
            input_floats,
            input_ints,
            input_mat_idcs,
            input_temp_mats,
            output_rows,
            output_cols,
            work_sizes
        )?;

        Ok(output_matrix)
    }
}

impl QuickCalculator<f64> {
    /// Initializes Calculator struct
    pub fn init() -> Result<QuickCalculator<f64>> {
        // Initialize vector of kernel names
        let program_vec: Vec<&str> = super::PROGRAM_LIST_DOUBLE.to_vec();
        
        // Create memory handler using program source and kernel names
        let memory_handler: MemoryHandler<f64> = MemoryHandler::<f64>::init(super::PROGRAM_SOURCE_DOUBLE, program_vec)?;

        // Create empty memory vector
        let mat_vector: Vec<Matrix<f64>> = Vec::with_capacity(super::INIT_MEMORY_CAPACITY);

        let customs_vector: Vec<ResultFunction<f64>> = Vec::default();
        let params_customs_vector: Vec<QuickParameterFunction<f64>> = Vec::default();
        let custom_idcs_vector: Vec<usize> = Vec::default();

        // Create and return new memory calculator
        let output: QuickCalculator<f64> = QuickCalculator {
            memory_handler: memory_handler,
            matrices: mat_vector,
            customs: customs_vector,
            params_customs: params_customs_vector,
            custom_idcs: custom_idcs_vector
        };
        Ok(output)
    }

    /// Multiply previously stored Matrix and previously stored Matrix
    pub fn mat_mul(&mut self, left_idx: usize, right_idx: usize) -> Result<Matrix<f64>> {
        let left: &Matrix<f64> = &self.matrices[left_idx];
        let right: &Matrix<f64> = &self.matrices[right_idx];

        if left.get_cols() != right.get_rows() {
            return Err(Jeeperr::DimensionError)
        }

        let rows: usize = left.get_rows();
        let interm: usize = left.get_cols();
        let cols: usize = right.get_cols();

        let output: Matrix<f64> = self.memory_handler.execute_once_and_read(
            0,
            None,
            Some(vec![cols as i32, interm as i32]),
            Some(vec![left_idx, right_idx]),
            None,
            rows,
            cols,
            vec![rows, cols]
        )?;

        Ok(output)
    }

    pub fn halfquick_mat_mul(&mut self, left_idx: usize, right: &Matrix<f64>) -> Result<Matrix<f64>> {
        let left: &Matrix<f64> = &self.matrices[left_idx];

        if left.get_cols() != right.get_rows() {
            return Err(Jeeperr::DimensionError)
        }

        let rows: usize = left.get_rows();
        let interm: usize = left.get_cols();
        let cols: usize = right.get_cols();

        let output: Matrix<f64> = self.memory_handler.execute_once_and_read(
            0,
            None,
            Some(vec![cols as i32, interm as i32]),
            Some(vec![left_idx]),
            Some(vec![right]),
            rows,
            cols,
            vec![rows, cols]
        )?;

        Ok(output)
    }

    /// Multiply non-stored Matrix and non-stored Matrix
    pub fn quick_mat_mul(&mut self, left: &Matrix<f64>, right: &Matrix<f64>) -> Result<Matrix<f64>> {
        if left.get_cols() != right.get_rows() {
            return Err(Jeeperr::DimensionError)
        }

        let rows: usize = left.get_rows();
        let interm: usize = left.get_cols();
        let cols: usize = right.get_cols();

        let output: Matrix<f64> = self.memory_handler.execute_once_and_read(
            0,
            None,
            Some(vec![cols as i32, interm as i32]),
            None,
            Some(vec![left, right]),
            rows,
            cols,
            vec![rows, cols]
        )?;

        Ok(output)
    }

    /// Compile and store a custom kernel and build/store a closure to execute said kernel
    pub unsafe fn load_custom_fn(
        &mut self,
        program_source: &str,
        kernel_name: &str,
        parameter_fn: QuickParameterFunction<f64>
    ) -> Result<usize>
    {
        let new_kernel_index: usize = self.memory_handler.new_kernel(program_source, kernel_name)?;
        self.custom_idcs.push(new_kernel_index);

        self.params_customs.push(parameter_fn);

        let new_custom: ResultFunction<f64> = Box::new(|
                index: usize,
                host: &mut MemoryHandler<f64>,
                input_floats: Option<Vec<f64>>,
                input_ints: Option<Vec<i32>>,
                input_mat_idcs: Option<Vec<usize>>,
                input_temp_mats: Option<Vec<&Matrix<f64>>>,
                output_rows: usize,
                output_cols: usize,
                work_sizes: Vec<usize>
            | -> Result<Matrix<f64>>
            {
                let output = host.execute_once_and_read(
                    index,
                    input_floats,
                    input_ints,
                    input_mat_idcs,
                    input_temp_mats,
                    output_rows,
                    output_cols,
                    work_sizes
                )?;

                Ok(output)
            }
        );

        self.customs.push(new_custom);

        return Ok(self.customs.len() - 1);
    }

    /// Execute custom kernel via pre-generated closure
    pub unsafe fn exec_custom_fn(
        &mut self,
        custom_index: usize,
        input_floats: Option<Vec<f64>>,
        input_ints: Option<Vec<i32>>,
        input_mat_idcs: Option<Vec<usize>>,
        input_temp_mats: Option<Vec<&Matrix<f64>>>
    ) -> Result<Matrix<f64>>
    {
        let input_matrices: Option<Vec<&Matrix<f64>>>;
        if input_mat_idcs.is_some() {
            input_matrices = Some(input_mat_idcs.clone()
                .unwrap()
                .clone()
                .iter()
                .map(| &idx | &self.matrices[idx])
                .collect::<Vec<&Matrix<f64>>>());
        }
        else {
            input_matrices = None;
        }

        let (output_rows, output_cols, work_sizes) = self.params_customs[custom_index](input_matrices.clone(), input_temp_mats.clone())?;

        let output_matrix = self.customs[custom_index](
            self.custom_idcs[custom_index],
            &mut self.memory_handler,
            input_floats,
            input_ints,
            input_mat_idcs,
            input_temp_mats,
            output_rows,
            output_cols,
            work_sizes
        )?;

        Ok(output_matrix)
    }
}