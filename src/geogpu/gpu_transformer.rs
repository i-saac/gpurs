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
    CL_MEM_WRITE_ONLY
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
use crate::Jeeperr;
use crate::geo::{
    Vec3h,
    Transform3h
};

/// GPU Memory Manager for Mass Vector Transformations.
pub struct GPUTransformer<T: IsFloat> {
    context: Context, // OpenCL context
    command_queue: CommandQueue, // OpenCL command queue
    kernels: Vec<Kernel>, // Vector of all compiled kernels
    write_buffers_3h: Vec<Buffer<T>>, // Vector of full write buffers
    last_write_event: Option<Event>, // Last write event object
    transforms_3h: Vec<Transform3h<T>> // Vector of loaded transforms
}

impl GPUTransformer<f32> {
    /// Initialize GPUTransformer.
    pub fn init() -> Result<GPUTransformer<f32>> {
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
            super::PROGRAM_SOURCE_FLOAT,
            ""
        ).expect("Failed to build program");
    
        // Initialize empty kernel vector
        let mut kernel_vector: Vec<Kernel> = Vec::with_capacity(super::PROGRAM_LIST_FLOAT.len());
    
        // Loop through each kernel name provided, create kernel and push to storage vector
        for kernel_name in super::PROGRAM_LIST_FLOAT {
            let kernel: Kernel = Kernel::create(&program, kernel_name)?;
    
            kernel_vector.push(kernel);
        }
    
        // Create empty buffer vector
        let buffer_vector: Vec<Buffer<f32>> = Vec::with_capacity(super::INIT_MEMORY_CAPACITY);

        // Create empty transform vector
        let stored_3h_vector: Vec<Transform3h<f32>> = Vec::default();
    
        // Create and return new GPUTrasnformer
        let output: GPUTransformer<f32> = GPUTransformer {
            context: context,
            command_queue: queue,
            kernels: kernel_vector,
            write_buffers_3h: buffer_vector,
            last_write_event: None,
            transforms_3h: stored_3h_vector,
        };
        Ok(output)
    }

    /// Store 3D Homogeneous transform.
    pub fn store_transform3h(&mut self, transform: Transform3h<f32>) -> Result<usize> {
        self.transforms_3h.push(transform.clone());

        let mut new_write_buffer: Buffer<f32> = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                16,
                ptr::null_mut()
            )?
        };

        self.last_write_event = Some(unsafe {
            self.command_queue.enqueue_write_buffer(
                &mut new_write_buffer,
                CL_NON_BLOCKING,
                0,
                &transform.get_data(),
                &[]
            )?
        });

        self.write_buffers_3h.push(new_write_buffer);

        let output_idx: usize = self.write_buffers_3h.len() - 1;
        if output_idx != (self.transforms_3h.len() - 1) {
            return Err(Jeeperr::MemoryError)
        }
        Ok(output_idx)
    }

    /// Mass transform 3D Homogeneous vectors.
    pub fn mass_transform_3h(&mut self, transform_idx: usize, inputs: &[Vec3h<f32>]) -> Result<Vec<Vec3h<f32>>> {
        let kernel_idx: usize = 0;

        let n_inputs: usize = inputs.len();

        let read_buffer: Buffer<Vec3h<f32>> = unsafe {
            Buffer::<Vec3h<f32>>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                n_inputs,
                ptr::null_mut()
            )?
        };

        let mut exec_kernel: ExecuteKernel = ExecuteKernel::new(&self.kernels[kernel_idx]);
        let mut events: Vec<cl_event> = Vec::default();

        let mut output_data: Vec<Vec3h<f32>> = vec![Vec3h::<f32>::i(); n_inputs];

        let mut kernel_mid_exec: &mut ExecuteKernel = unsafe {
            exec_kernel.set_arg(&read_buffer)
        };

        kernel_mid_exec = unsafe {
            kernel_mid_exec.set_arg(&(n_inputs as i32))
        };

        kernel_mid_exec = unsafe {
            kernel_mid_exec.set_arg(&self.write_buffers_3h[transform_idx])
        };

        let mut temp_write_buffer: Buffer<Vec3h<f32>> = unsafe {
            Buffer::<Vec3h<f32>>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                n_inputs,
                ptr::null_mut()
            )?
        };

        // Write matrix to buffer and store event
        let last_write_event: Event = unsafe {
            self.command_queue.enqueue_write_buffer(
                &mut temp_write_buffer,
                CL_NON_BLOCKING,
                0,
                &inputs,
                &[]
            )?
        };

        kernel_mid_exec = unsafe {
            kernel_mid_exec.set_arg(&temp_write_buffer)
        };

        let kernel_event: Event = unsafe {
            kernel_mid_exec
                .set_global_work_sizes(&[n_inputs])
                .set_wait_event(&last_write_event)
                .enqueue_nd_range(&self.command_queue)?
        };

        events.push(kernel_event.get());

        let read_event: Event = unsafe {
            self.command_queue.enqueue_read_buffer(
                &read_buffer,
                CL_NON_BLOCKING,
                0,
                &mut output_data,
                &events
            )?
        };

        read_event.wait()?;

        self.last_write_event = Some(kernel_event);

        Ok(output_data)
    }
}


impl GPUTransformer<f64> {
    /// Initialize GPUTransformer.
    pub fn init() -> Result<GPUTransformer<f64>> {
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
            super::PROGRAM_SOURCE_DOUBLE,
            ""
        ).expect("Failed to build program");
    
        // Initialize empty kernel vector
        let mut kernel_vector: Vec<Kernel> = Vec::with_capacity(super::PROGRAM_LIST_DOUBLE.len());
    
        // Loop through each kernel name provided, create kernel and push to storage vector
        for kernel_name in super::PROGRAM_LIST_DOUBLE {
            let kernel: Kernel = Kernel::create(&program, kernel_name)?;
    
            kernel_vector.push(kernel);
        }
    
        // Create empty buffer vector
        let buffer_vector: Vec<Buffer<f64>> = Vec::with_capacity(super::INIT_MEMORY_CAPACITY);

        // Create empty transforma vector
        let stored_3h_vector: Vec<Transform3h<f64>> = Vec::default();
    
        // Create and return new GPUTransformer
        let output: GPUTransformer<f64> = GPUTransformer {
            context: context,
            command_queue: queue,
            kernels: kernel_vector,
            write_buffers_3h: buffer_vector,
            last_write_event: None,
            transforms_3h: stored_3h_vector,
        };
        Ok(output)
    }
    
    /// Store 3D Homogeneous transform.
    pub fn store_transform3h(&mut self, transform: Transform3h<f64>) -> Result<usize> {
        self.transforms_3h.push(transform.clone());

        let mut new_write_buffer: Buffer<f64> = unsafe {
            Buffer::<f64>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                16,
                ptr::null_mut()
            )?
        };

        self.last_write_event = Some(unsafe {
            self.command_queue.enqueue_write_buffer(
                &mut new_write_buffer,
                CL_NON_BLOCKING,
                0,
                &transform.get_data(),
                &[]
            )?
        });

        self.write_buffers_3h.push(new_write_buffer);

        let output_idx: usize = self.write_buffers_3h.len() - 1;
        if output_idx != (self.transforms_3h.len() - 1) {
            return Err(Jeeperr::MemoryError)
        }
        Ok(output_idx)
    }

    /// Mass transform 3D Homogeneous vectors.
    pub fn mass_transform_3h(&mut self, transform_idx: usize, inputs: &[Vec3h<f64>]) -> Result<Vec<Vec3h<f64>>> {
        let kernel_idx: usize = 0;

        let n_inputs: usize = inputs.len();

        let read_buffer: Buffer<Vec3h<f64>> = unsafe {
            Buffer::<Vec3h<f64>>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                n_inputs,
                ptr::null_mut()
            )?
        };

        let mut exec_kernel: ExecuteKernel = ExecuteKernel::new(&self.kernels[kernel_idx]);
        let mut events: Vec<cl_event> = Vec::default();

        let mut output_data: Vec<Vec3h<f64>> = vec![Vec3h::<f64>::i(); n_inputs];

        let mut kernel_mid_exec: &mut ExecuteKernel = unsafe {
            exec_kernel.set_arg(&read_buffer)
        };

        kernel_mid_exec = unsafe {
            kernel_mid_exec.set_arg(&(n_inputs as i32))
        };

        kernel_mid_exec = unsafe {
            kernel_mid_exec.set_arg(&self.write_buffers_3h[transform_idx])
        };

        let mut temp_write_buffer: Buffer<Vec3h<f64>> = unsafe {
            Buffer::<Vec3h<f64>>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                n_inputs,
                ptr::null_mut()
            )?
        };

        // Write matrix to buffer and store event
        let last_write_event: Event = unsafe {
            self.command_queue.enqueue_write_buffer(
                &mut temp_write_buffer,
                CL_NON_BLOCKING,
                0,
                &inputs,
                &[]
            )?
        };

        kernel_mid_exec = unsafe {
            kernel_mid_exec.set_arg(&temp_write_buffer)
        };

        let kernel_event: Event = unsafe {
            kernel_mid_exec
                .set_global_work_sizes(&[n_inputs])
                .set_wait_event(&last_write_event)
                .enqueue_nd_range(&self.command_queue)?
        };

        events.push(kernel_event.get());

        let read_event: Event = unsafe {
            self.command_queue.enqueue_read_buffer(
                &read_buffer,
                CL_NON_BLOCKING,
                0,
                &mut output_data,
                &events
            )?
        };

        read_event.wait()?;

        self.last_write_event = Some(kernel_event);

        Ok(output_data)
    }
}