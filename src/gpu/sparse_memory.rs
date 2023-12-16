//! Behind-the-scenes brain of the gpu module.
//! Handles GPU memory management for MemoryCalculator and QuickCalculator

extern crate opencl3;

use opencl3::command_queue::{
    CommandQueue,
    CL_QUEUE_PROFILING_ENABLE
};
use opencl3::context::Context;
use opencl3::device::{
    get_all_devices,
    Device,
    CL_DEVICE_TYPE_GPU
};
use opencl3::kernel::{
    ExecuteKernel,
    Kernel
};
use opencl3::memory::{
    Buffer,
    CL_MEM_READ_ONLY,
    CL_MEM_READ_WRITE
};
use opencl3::program::Program;
use opencl3::types::{
    cl_event,
    CL_NON_BLOCKING
};
use opencl3::event::Event;
use std::ptr;

use crate::IsFloat;
use crate::Result;

use crate::linalg::Matrix;
use crate::linalg::SparseMatrix;

pub struct SparseMemoryHandler<T: IsFloat + std::fmt::Debug + Copy + Clone> {
    context: Context, // OpenCL context
    command_queue: CommandQueue, // OpenCL command queue
    kernels: Vec<Kernel>, // Vector of all compiled kernels
    sparse_write_buffers: Vec<(Buffer<i32>, Buffer<i32>, Buffer<T>)>, // Vector of full sparse write buffers
    dense_write_buffers: Vec<Buffer<T>>, // Vector of full dense write buffers
    last_write_event: Option<Event> // Last write event object
}

impl<T: IsFloat + std::fmt::Debug + Copy + Clone> SparseMemoryHandler<T> {
    // Create buffer and store sparse matrix in buffer
    pub fn store_sparse_matrix(&mut self, matrix: SparseMatrix<T>) -> Result<usize> {
        let (row_starts, col_indices, values) = matrix.to_csr(false);

        // Create new empty write buffer
        let mut row_write_buffer: Buffer<i32> = unsafe {
            Buffer::<i32>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                row_starts.len(),
                ptr::null_mut()
            )?
        };
        
        let mut col_write_buffer: Buffer<i32> = unsafe {
            Buffer::<i32>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                col_indices.len(),
                ptr::null_mut()
            )?
        };

        let mut val_write_buffer: Buffer<T> = unsafe {
            Buffer::<T>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                values.len(),
                ptr::null_mut()
            )?
        };

        self.last_write_event = Some(unsafe {
            self.command_queue.enqueue_write_buffer(
                &mut row_write_buffer,
                CL_NON_BLOCKING,
                0,
                &row_starts.into_iter().map(|i| i as i32).collect::<Vec<i32>>(),
                &[]
            )?;

            self.command_queue.enqueue_write_buffer(
                &mut col_write_buffer,
                CL_NON_BLOCKING,
                0,
                &col_indices.into_iter().map(|i| i as i32).collect::<Vec<i32>>(),
                &[]
            )?;

            self.command_queue.enqueue_write_buffer(
                &mut val_write_buffer,
                CL_NON_BLOCKING,
                0,
                &values,
                &[]
            )?
        });

        // Store full write buffer to handler memory
        self.sparse_write_buffers.push((row_write_buffer, col_write_buffer, val_write_buffer));

        // Return index of new write buffer
        let output: usize = self.sparse_write_buffers.len() - 1;
        Ok(output)
    }

    // Create buffer and store dense matrix in buffer
    pub fn store_dense_matrix(&mut self, matrix: Matrix<T>) -> Result<usize> {
        // Create new empty write buffer
        let mut new_write_buffer: Buffer<T> = unsafe {
            Buffer::<T>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                matrix.get_rows() * matrix.get_cols(),
                ptr::null_mut()
            )?
        };

        // Write matrix to buffer and store event
        self.last_write_event = Some(unsafe {
            self.command_queue.enqueue_write_buffer(
                &mut new_write_buffer,
                CL_NON_BLOCKING,
                0,
                &matrix.get_data(),
                &[]
            )?
        });

        // Store full write buffer to handler memory
        self.dense_write_buffers.push(new_write_buffer);

        // Return index of new write buffer
        let output: usize = self.dense_write_buffers.len() - 1;
        Ok(output)
    }

    pub fn new_kernel(&mut self, program_source: &str, kernel_name: &str) -> Result<usize> {
        // Compile program from source
        let program: Program = Program::create_and_build_from_source(
            &self.context,
            program_source,
            ""
        ).expect("Failed to build program");

        let kernel: Kernel = Kernel::create(&program, kernel_name)?;

        self.kernels.push(kernel);

        Ok(self.kernels.len() - 1)
    }
}

impl SparseMemoryHandler<f32> {
    pub fn init(program_source: &str, kernel_names: Vec<&str>) -> Result<SparseMemoryHandler<f32>> {
        // Get devices and create device object
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
            .first()
            .expect("No device found in platform");
        let device: Device = Device::new(device_id);
    
        // Create context object from device
        let context: Context = Context::from_device(&device)?;
    
        // Create command queue from context with default queue size
        let queue: CommandQueue = CommandQueue::create_default_with_properties(
            &context,
            CL_QUEUE_PROFILING_ENABLE,
            0
        )?;
    
        // Compile program from source
        let program: Program = Program::create_and_build_from_source(
            &context,
            program_source,
            ""
        ).expect("Failed to build program");
    
        // Initialize empty kernel vector
        let mut kernel_vector: Vec<Kernel> = Vec::with_capacity(kernel_names.len());
    
        // Loop through each kernel name provided, create kernel and push to storage vector
        for kernel_name in kernel_names {
            let kernel: Kernel = Kernel::create(&program, kernel_name)?;
    
            kernel_vector.push(kernel);
        }
    
        // Create empty buffer vector
        let sparse_buffer_vector: Vec<(Buffer<i32>, Buffer<i32>, Buffer<f32>)> = Vec::with_capacity(super::INIT_MEMORY_CAPACITY);
        let dense_buffer_vector: Vec<Buffer<f32>> = Vec::with_capacity(super::INIT_MEMORY_CAPACITY);
    
        // Create and return new Memory Handler
        let output: SparseMemoryHandler<f32> = SparseMemoryHandler {
            context: context,
            command_queue: queue,
            kernels: kernel_vector,
            sparse_write_buffers: sparse_buffer_vector,
            dense_write_buffers: dense_buffer_vector,
            last_write_event: None
        };
        Ok(output)
    }

    // Execute kernel and return resulting matrix
    pub fn execute_and_read(
        &mut self,
        kernel_index: usize,
        input_floats: Option<Vec<f32>>,
        input_ints: Option<Vec<i32>>,
        input_sparse_mat_idcs: Vec<usize>,
        input_dense_mat_idcs: Vec<usize>,
        output_rows: usize,
        output_cols: usize,
        work_sizes: Vec<usize>
    ) -> Result<(Matrix<f32>, usize)>
    {
        // Create read buffer
        let read_buffer: Buffer<f32> = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                output_rows * output_cols,
                ptr::null_mut()
            )?
        };

        // Create ExecuteKernel object and empty events vector
        let mut exec_kernel: ExecuteKernel = ExecuteKernel::new(&self.kernels[kernel_index]);
        let mut events: Vec<cl_event> = Vec::default();

        // Create output vector and initilize with zeros
        let mut output_data: Vec<f32> = vec![0.0; output_rows * output_cols];

        // Give ExecuteKernel the read buffer
        let mut kernel_mid_exec: &mut ExecuteKernel = unsafe {
            exec_kernel.set_arg(&read_buffer)
        };

        // Give ExecuteKernel the provided floats
        if input_floats.is_some() {
            for write_value in input_floats.unwrap() {
                kernel_mid_exec = unsafe {
                    kernel_mid_exec.set_arg(&write_value)
                }
            }
        }

        // Give ExecuteKernel the provided ints
        if input_ints.is_some() {
            for write_value in input_ints.unwrap() {
                kernel_mid_exec = unsafe {
                    kernel_mid_exec.set_arg(&write_value)
                }
            }
        }

        let selected_sparse_write_buffers: Vec<&(Buffer<i32>, Buffer<i32>, Buffer<f32>)> = input_sparse_mat_idcs
            .iter()
            .map(|&idx| &self.sparse_write_buffers[idx])
            .collect::<Vec<_>>();

        for (selected_row_write_buffer, selected_col_write_buffer, selected_val_write_buffer) in selected_sparse_write_buffers {
            kernel_mid_exec = unsafe {
                kernel_mid_exec.set_arg(&selected_row_write_buffer)
            };
            kernel_mid_exec = unsafe {
                kernel_mid_exec.set_arg(&selected_col_write_buffer)
            };
            kernel_mid_exec = unsafe {
                kernel_mid_exec.set_arg(&selected_val_write_buffer)
            };
        }

        // Collect selected write buffers into separate vector
        let selected_dense_write_buffers: Vec<&Buffer<f32>> = input_dense_mat_idcs
            .iter()
            .map(|&idx| &self.dense_write_buffers[idx])
            .collect::<Vec<_>>();

        // Give ExecuteKernel the selected matrices
        for selected_buffer in selected_dense_write_buffers {
            kernel_mid_exec = unsafe {
                kernel_mid_exec.set_arg(selected_buffer)
            };
        }

        // Unwrap Option from before to reference later
        let last_write_event: &Event = self.last_write_event.as_ref().unwrap();

        // Finalize kernel execution and store final event
        let kernel_event: Event = unsafe {
            kernel_mid_exec
                .set_global_work_sizes(&work_sizes)
                .set_wait_event(last_write_event)
                .enqueue_nd_range(&self.command_queue)?
        };

        // Store raw kernel event in events vector
        events.push(kernel_event.get());

        // Read from buffer and store in output vector
        let read_event: Event = unsafe {
            self.command_queue.enqueue_read_buffer(
                &read_buffer,
                CL_NON_BLOCKING,
                0,
                &mut output_data,
                &events
            )?
        };

        // Wait for read event to finish
        read_event.wait()?;

        // Store new calculated buffer in handler memory
        self.dense_write_buffers.push(read_buffer);

        // Update most recent write event
        self.last_write_event = Some(kernel_event);

        // Create and return output matrix and memory index
        let output: Matrix<f32> = Matrix::new(output_data, output_rows, output_cols)?;
        let output_idx: usize = self.dense_write_buffers.len() - 1;
        Ok((output, output_idx))
    }

    pub fn execute_once_and_read(
        &mut self,
        kernel_index: usize,
        input_floats: Option<Vec<f32>>,
        input_ints: Option<Vec<i32>>,
        input_sparse_mat_idcs: Vec<usize>,
        input_dense_mat_idcs: Option<Vec<usize>>,
        input_temp_mats: Option<Vec<&Matrix<f32>>>,
        output_rows: usize,
        output_cols: usize,
        work_sizes: Vec<usize>
    ) -> Result<Matrix<f32>>
    {
        // Create read buffer
        let read_buffer: Buffer<f32> = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                output_rows * output_cols,
                ptr::null_mut()
            )?
        };

        // Create ExecuteKernel object and empty events vector
        let mut exec_kernel: ExecuteKernel = ExecuteKernel::new(&self.kernels[kernel_index]);
        let mut events: Vec<cl_event> = Vec::default();

        // Create output vector and initilize with zeros
        let mut output_data: Vec<f32> = vec![0.0; output_rows * output_cols];

        // Give ExecuteKernel the read buffer
        let mut kernel_mid_exec: &mut ExecuteKernel = unsafe {
            exec_kernel.set_arg(&read_buffer)
        };

        // Give ExecuteKernel the provided floats
        if input_floats.is_some() {
            for write_value in input_floats.unwrap() {
                kernel_mid_exec = unsafe {
                    kernel_mid_exec.set_arg(&write_value)
                }
            }
        }

        // Give ExecuteKernel the provided ints
        if input_ints.is_some() {
            for write_value in input_ints.unwrap() {
                kernel_mid_exec = unsafe {
                    kernel_mid_exec.set_arg(&write_value)
                }
            }
        }

        let selected_sparse_write_buffers: Vec<&(Buffer<i32>, Buffer<i32>, Buffer<f32>)> = input_sparse_mat_idcs
            .iter()
            .map(|&idx| &self.sparse_write_buffers[idx])
            .collect::<Vec<_>>();

        for (selected_row_write_buffer, selected_col_write_buffer, selected_val_write_buffer) in selected_sparse_write_buffers {
            kernel_mid_exec = unsafe {
                kernel_mid_exec.set_arg(&selected_row_write_buffer)
            };
            kernel_mid_exec = unsafe {
                kernel_mid_exec.set_arg(&selected_col_write_buffer)
            };
            kernel_mid_exec = unsafe {
                kernel_mid_exec.set_arg(&selected_val_write_buffer)
            };
        }

        if input_dense_mat_idcs.is_some() {
            // Collect selected write buffers into separate vector
            let selected_write_buffers: Vec<&Buffer<f32>> = input_dense_mat_idcs
                .unwrap()
                .iter()
                .map(|&idx| &self.dense_write_buffers[idx])
                .collect::<Vec<_>>();

            // Give ExecuteKernel the selected matrices
            for selected_buffer in selected_write_buffers {
                kernel_mid_exec = unsafe {
                    kernel_mid_exec.set_arg(selected_buffer)
                };
            }
        }

        let mut temp_write_buffers: Vec<Buffer<f32>> = Vec::default();
        if input_temp_mats.is_some() {
            // Loop through temp matrix entries
            for temp_matrix in input_temp_mats.unwrap() {
                // Create temporary write buffer
                let mut temp_write_buffer: Buffer<f32> = unsafe {
                    Buffer::<f32>::create(
                        &self.context,
                        CL_MEM_READ_ONLY,
                        temp_matrix.get_rows() * temp_matrix.get_cols(),
                        ptr::null_mut()
                    )?
                };

                // Write matrix to buffer and store event
                self.last_write_event = Some(unsafe {
                    self.command_queue.enqueue_write_buffer(
                        &mut temp_write_buffer,
                        CL_NON_BLOCKING,
                        0,
                        &temp_matrix.get_data(),
                        &[]
                    )?
                });

                // Give ExecuteKernel the new buffer
                kernel_mid_exec = unsafe {
                    kernel_mid_exec.set_arg(&temp_write_buffer)
                };
                
                temp_write_buffers.push(temp_write_buffer);
            }               
        }

        // Unwrap Option from before to reference later
        let last_write_event: &Event = self.last_write_event.as_ref().unwrap();

        // Finalize kernel execution and store final event
        let kernel_event: Event = unsafe {
            kernel_mid_exec
                .set_global_work_sizes(&work_sizes)
                .set_wait_event(last_write_event)
                .enqueue_nd_range(&self.command_queue)?
        };

        // Store raw kernel event in events vector
        events.push(kernel_event.get());

        // Read from buffer and store in output vector
        let read_event: Event = unsafe {
            self.command_queue.enqueue_read_buffer(
                &read_buffer,
                CL_NON_BLOCKING,
                0,
                &mut output_data,
                &events
            )?
        };

        // Wait for read event to finish
        read_event.wait()?;

        // Update most recent write event
        self.last_write_event = Some(kernel_event);

        // Create and return output matrix and memory index
        let output: Matrix<f32> = Matrix::new(output_data, output_rows, output_cols)?;
        Ok(output)
    }
}

impl SparseMemoryHandler<f64> {
    pub fn init(program_source: &str, kernel_names: Vec<&str>) -> Result<SparseMemoryHandler<f64>> {
        // Get devices and create device object
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
            .first()
            .expect("No device found in platform");
        let device: Device = Device::new(device_id);
    
        // Create context object from device
        let context: Context = Context::from_device(&device)?;
    
        // Create command queue from context with default queue size
        let queue: CommandQueue = CommandQueue::create_default_with_properties(
            &context,
            CL_QUEUE_PROFILING_ENABLE,
            0
        )?;
    
        // Compile program from source
        let program: Program = Program::create_and_build_from_source(
            &context,
            program_source,
            ""
        ).expect("Failed to build program");
    
        // Initialize empty kernel vector
        let mut kernel_vector: Vec<Kernel> = Vec::with_capacity(kernel_names.len());
    
        // Loop through each kernel name provided, create kernel and push to storage vector
        for kernel_name in kernel_names {
            let kernel: Kernel = Kernel::create(&program, kernel_name)?;
    
            kernel_vector.push(kernel);
        }
    
        // Create empty buffer vector
        let sparse_buffer_vector: Vec<(Buffer<i32>, Buffer<i32>, Buffer<f64>)> = Vec::with_capacity(super::INIT_MEMORY_CAPACITY);
        let dense_buffer_vector: Vec<Buffer<f64>> = Vec::with_capacity(super::INIT_MEMORY_CAPACITY);
    
        // Create and return new Memory Handler
        let output: SparseMemoryHandler<f64> = SparseMemoryHandler {
            context: context,
            command_queue: queue,
            kernels: kernel_vector,
            sparse_write_buffers: sparse_buffer_vector,
            dense_write_buffers: dense_buffer_vector,
            last_write_event: None
        };
        Ok(output)
    }

     // Execute kernel and return resulting matrix
     pub fn execute_and_read(
        &mut self,
        kernel_index: usize,
        input_floats: Option<Vec<f64>>,
        input_ints: Option<Vec<i32>>,
        input_sparse_mat_idcs: Vec<usize>,
        input_dense_mat_idcs: Vec<usize>,
        output_rows: usize,
        output_cols: usize,
        work_sizes: Vec<usize>
    ) -> Result<(Matrix<f64>, usize)>
    {
        // Create read buffer
        let read_buffer: Buffer<f64> = unsafe {
            Buffer::<f64>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                output_rows * output_cols,
                ptr::null_mut()
            )?
        };

        // Create ExecuteKernel object and empty events vector
        let mut exec_kernel: ExecuteKernel = ExecuteKernel::new(&self.kernels[kernel_index]);
        let mut events: Vec<cl_event> = Vec::default();

        // Create output vector and initilize with zeros
        let mut output_data: Vec<f64> = vec![0.0; output_rows * output_cols];

        // Give ExecuteKernel the read buffer
        let mut kernel_mid_exec: &mut ExecuteKernel = unsafe {
            exec_kernel.set_arg(&read_buffer)
        };

        // Give ExecuteKernel the provided floats
        if input_floats.is_some() {
            for write_value in input_floats.unwrap() {
                kernel_mid_exec = unsafe {
                    kernel_mid_exec.set_arg(&write_value)
                }
            }
        }

        // Give ExecuteKernel the provided ints
        if input_ints.is_some() {
            for write_value in input_ints.unwrap() {
                kernel_mid_exec = unsafe {
                    kernel_mid_exec.set_arg(&write_value)
                }
            }
        }

        let selected_sparse_write_buffers: Vec<&(Buffer<i32>, Buffer<i32>, Buffer<f64>)> = input_sparse_mat_idcs
            .iter()
            .map(|&idx| &self.sparse_write_buffers[idx])
            .collect::<Vec<_>>();

        for (selected_row_write_buffer, selected_col_write_buffer, selected_val_write_buffer) in selected_sparse_write_buffers {
            kernel_mid_exec = unsafe {
                kernel_mid_exec.set_arg(&selected_row_write_buffer)
            };
            kernel_mid_exec = unsafe {
                kernel_mid_exec.set_arg(&selected_col_write_buffer)
            };
            kernel_mid_exec = unsafe {
                kernel_mid_exec.set_arg(&selected_val_write_buffer)
            };
        }

        // Collect selected write buffers into separate vector
        let selected_dense_write_buffers: Vec<&Buffer<f64>> = input_dense_mat_idcs
            .iter()
            .map(|&idx| &self.dense_write_buffers[idx])
            .collect::<Vec<_>>();

        // Give ExecuteKernel the selected matrices
        for selected_buffer in selected_dense_write_buffers {
            kernel_mid_exec = unsafe {
                kernel_mid_exec.set_arg(selected_buffer)
            };
        }

        // Unwrap Option from before to reference later
        let last_write_event: &Event = self.last_write_event.as_ref().unwrap();

        // Finalize kernel execution and store final event
        let kernel_event: Event = unsafe {
            kernel_mid_exec
                .set_global_work_sizes(&work_sizes)
                .set_wait_event(last_write_event)
                .enqueue_nd_range(&self.command_queue)?
        };

        // Store raw kernel event in events vector
        events.push(kernel_event.get());

        // Read from buffer and store in output vector
        let read_event: Event = unsafe {
            self.command_queue.enqueue_read_buffer(
                &read_buffer,
                CL_NON_BLOCKING,
                0,
                &mut output_data,
                &events
            )?
        };

        // Wait for read event to finish
        read_event.wait()?;

        // Store new calculated buffer in handler memory
        self.dense_write_buffers.push(read_buffer);

        // Update most recent write event
        self.last_write_event = Some(kernel_event);

        // Create and return output matrix and memory index
        let output: Matrix<f64> = Matrix::new(output_data, output_rows, output_cols)?;
        let output_idx: usize = self.dense_write_buffers.len() - 1;
        Ok((output, output_idx))
    }

    pub fn execute_once_and_read(
        &mut self,
        kernel_index: usize,
        input_floats: Option<Vec<f64>>,
        input_ints: Option<Vec<i32>>,
        input_sparse_mat_idcs: Vec<usize>,
        input_dense_mat_idcs: Option<Vec<usize>>,
        input_temp_mats: Option<Vec<&Matrix<f64>>>,
        output_rows: usize,
        output_cols: usize,
        work_sizes: Vec<usize>
    ) -> Result<Matrix<f64>>
    {
        // Create read buffer
        let read_buffer: Buffer<f64> = unsafe {
            Buffer::<f64>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                output_rows * output_cols,
                ptr::null_mut()
            )?
        };

        // Create ExecuteKernel object and empty events vector
        let mut exec_kernel: ExecuteKernel = ExecuteKernel::new(&self.kernels[kernel_index]);
        let mut events: Vec<cl_event> = Vec::default();

        // Create output vector and initilize with zeros
        let mut output_data: Vec<f64> = vec![0.0; output_rows * output_cols];

        // Give ExecuteKernel the read buffer
        let mut kernel_mid_exec: &mut ExecuteKernel = unsafe {
            exec_kernel.set_arg(&read_buffer)
        };

        // Give ExecuteKernel the provided floats
        if input_floats.is_some() {
            for write_value in input_floats.unwrap() {
                kernel_mid_exec = unsafe {
                    kernel_mid_exec.set_arg(&write_value)
                }
            }
        }

        // Give ExecuteKernel the provided ints
        if input_ints.is_some() {
            for write_value in input_ints.unwrap() {
                kernel_mid_exec = unsafe {
                    kernel_mid_exec.set_arg(&write_value)
                }
            }
        }

        let selected_sparse_write_buffers: Vec<&(Buffer<i32>, Buffer<i32>, Buffer<f64>)> = input_sparse_mat_idcs
            .iter()
            .map(|&idx| &self.sparse_write_buffers[idx])
            .collect::<Vec<_>>();

        for (selected_row_write_buffer, selected_col_write_buffer, selected_val_write_buffer) in selected_sparse_write_buffers {
            kernel_mid_exec = unsafe {
                kernel_mid_exec.set_arg(selected_row_write_buffer)
            };
            kernel_mid_exec = unsafe {
                kernel_mid_exec.set_arg(selected_col_write_buffer)
            };
            kernel_mid_exec = unsafe {
                kernel_mid_exec.set_arg(selected_val_write_buffer)
            };
        }

        if input_dense_mat_idcs.is_some() {
            // Collect selected write buffers into separate vector
            let selected_write_buffers: Vec<&Buffer<f64>> = input_dense_mat_idcs
                .unwrap()
                .iter()
                .map(|&idx| &self.dense_write_buffers[idx])
                .collect::<Vec<_>>();

            // Give ExecuteKernel the selected matrices
            for selected_buffer in selected_write_buffers {
                kernel_mid_exec = unsafe {
                    kernel_mid_exec.set_arg(selected_buffer)
                };
            }
        }

        let mut temp_write_buffers: Vec<Buffer<f64>> = Vec::default();
        if input_temp_mats.is_some() {
            // Loop through temp matrix entries
            for temp_matrix in input_temp_mats.unwrap() {
                // Create temporary write buffer
                let mut temp_write_buffer: Buffer<f64> = unsafe {
                    Buffer::<f64>::create(
                        &self.context,
                        CL_MEM_READ_ONLY,
                        temp_matrix.get_rows() * temp_matrix.get_cols(),
                        ptr::null_mut()
                    )?
                };

                // Write matrix to buffer and store event
                self.last_write_event = Some(unsafe {
                    self.command_queue.enqueue_write_buffer(
                        &mut temp_write_buffer,
                        CL_NON_BLOCKING,
                        0,
                        &temp_matrix.get_data(),
                        &[]
                    )?
                });

                // Give ExecuteKernel the new buffer
                kernel_mid_exec = unsafe {
                    kernel_mid_exec.set_arg(&temp_write_buffer)
                };
                
                temp_write_buffers.push(temp_write_buffer);
            }               
        }

        // Unwrap Option from before to reference later
        let last_write_event: &Event = self.last_write_event.as_ref().unwrap();

        // Finalize kernel execution and store final event
        let kernel_event: Event = unsafe {
            kernel_mid_exec
                .set_global_work_sizes(&work_sizes)
                .set_wait_event(last_write_event)
                .enqueue_nd_range(&self.command_queue)?
        };

        // Store raw kernel event in events vector
        events.push(kernel_event.get());

        // Read from buffer and store in output vector
        let read_event: Event = unsafe {
            self.command_queue.enqueue_read_buffer(
                &read_buffer,
                CL_NON_BLOCKING,
                0,
                &mut output_data,
                &events
            )?
        };

        // Wait for read event to finish
        read_event.wait()?;

        // Update most recent write event
        self.last_write_event = Some(kernel_event);

        // Create and return output matrix and memory index
        let output: Matrix<f64> = Matrix::new(output_data, output_rows, output_cols)?;
        Ok(output)
    }
}