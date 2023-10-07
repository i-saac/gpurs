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

pub struct MemoryHandler {
    context: Context, // OpenCL context
    command_queue: CommandQueue, // OpenCL command queue
    kernels: Vec<Kernel>, // Vector of all compiled kernels
    write_buffers: Vec<Buffer<f32>>, // Vector of full write buffers
    last_write_event: Option<Event> // Last write event object
}

use crate::Result;

use crate::Jeeperr;
use crate::Matrix;

impl MemoryHandler {
    pub fn new(program_source: &str, kernel_names: Vec<&str>) -> Result<MemoryHandler> {
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
        let buffer_vector: Vec<Buffer<f32>> = Vec::with_capacity(super::INIT_MEMORY_CAPACITY);
    
        // Create and return new Memory Handler
        let output: MemoryHandler = MemoryHandler {
            context: context,
            command_queue: queue,
            kernels: kernel_vector,
            write_buffers: buffer_vector,
            last_write_event: None
        };
        Ok(output)
    }

    // Create buffer and store matrix in buffer
    pub fn store_matrix(&mut self, matrix: Matrix) -> Result<usize> {
        if matrix.get_rows() * matrix.get_cols() != matrix.get_data().len() {
            return Err(Jeeperr::DimensionError)
        }

        // Create new empty write buffer
        let mut new_write_buffer: Buffer<f32> = unsafe {
            Buffer::<f32>::create(
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
        self.write_buffers.push(new_write_buffer);

        // Return index of new write buffer
        let output: usize = self.write_buffers.len() - 1;
        Ok(output)
    }

    // Execute kernel and return resulting matrix
    pub fn execute_and_read(
        &mut self,
        kernel_index: usize,
        input_floats: Option<Vec<f32>>,
        input_ints: Option<Vec<i32>>,
        input_mat_idcs: Vec<usize>,
        output_rows: usize,
        output_cols: usize,
        work_sizes: Vec<usize>
    ) -> Result<(Matrix, usize)>
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

        // Collect selected write buffers into separate vector
        let selected_write_buffers: Vec<&Buffer<f32>> = input_mat_idcs
            .iter()
            .map(|idx| &self.write_buffers[*idx])
            .collect::<Vec<_>>();

        // Give ExecuteKernel the selected matrices
        for selected_buffer in selected_write_buffers {
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
        self.write_buffers.push(read_buffer);

        // Update most recent write event
        self.last_write_event = Some(kernel_event);

        // Create and return output matrix and memory index
        let output: Matrix = Matrix::new(output_data, output_rows, output_cols)?;
        let output_idx: usize = self.write_buffers.len() - 1;
        Ok((output, output_idx))
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