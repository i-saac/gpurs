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

use crate::Result;
use crate::Jeeperr;
use crate::geo::{
    Vec3h,
    Transform3h
};

pub struct GPUTransformer {
    context: Context, // OpenCL context
    command_queue: CommandQueue, // OpenCL command queue
    kernels: Vec<Kernel>, // Vector of all compiled kernels
    write_buffers_3h: Vec<Buffer<f32>>, // Vector of full write buffers
    last_write_event: Option<Event>, // Last write event object
    transforms_3h: Vec<Transform3h>
}

impl GPUTransformer {
    pub fn init() -> Result<GPUTransformer> {
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
            super::PROGRAM_SOURCE,
            ""
        ).expect("Failed to build program");
    
        // Initialize empty kernel vector
        let mut kernel_vector: Vec<Kernel> = Vec::with_capacity(super::PROGRAM_LIST.len());
    
        // Loop through each kernel name provided, create kernel and push to storage vector
        for kernel_name in super::PROGRAM_LIST {
            let kernel: Kernel = Kernel::create(&program, kernel_name)?;
    
            kernel_vector.push(kernel);
        }
    
        // Create empty buffer vector
        let buffer_vector: Vec<Buffer<f32>> = Vec::with_capacity(super::INIT_MEMORY_CAPACITY);
    
        // Create and return new Memory Handler
        let output: GPUTransformer = GPUTransformer {
            context: context,
            command_queue: queue,
            kernels: kernel_vector,
            write_buffers_3h: buffer_vector,
            last_write_event: None,
            transforms_3h: Vec::default(),
        };
        Ok(output)
    }

    pub fn store_transform3h(&mut self, transform: Transform3h) -> Result<usize> {
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

    pub fn mass_transform_3h(&mut self, transform_idx: usize, inputs: &[Vec3h]) -> Result<Vec<Vec3h>> {
        let kernel_idx: usize = 0;

        let n_inputs: usize = inputs.len();

        let read_buffer: Buffer<f32> = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                4 * n_inputs,
                ptr::null_mut()
            )?
        };

        let mut exec_kernel: ExecuteKernel = ExecuteKernel::new(&self.kernels[kernel_idx]);
        let mut events: Vec<cl_event> = Vec::default();

        let mut output_data: Vec<f32> = vec![0.0; 4 * n_inputs];

        let mut kernel_mid_exec: &mut ExecuteKernel = unsafe {
            exec_kernel.set_arg(&read_buffer)
        };

        kernel_mid_exec = unsafe {
            kernel_mid_exec.set_arg(&(n_inputs as i32))
        };

        kernel_mid_exec = unsafe {
            kernel_mid_exec.set_arg(&self.write_buffers_3h[transform_idx])
        };

        let mut temp_write_buffer: Buffer<f32> = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                4 * n_inputs,
                ptr::null_mut()
            )?
        };

        let input_vectors_flattened: Vec<f32> = (0..n_inputs).into_iter()
            .flat_map(|idx| [inputs[idx].x, inputs[idx].y, inputs[idx].z, inputs[idx].w])
            .collect::<Vec<f32>>();

        // Write matrix to buffer and store event
        let last_write_event: Event = unsafe {
            self.command_queue.enqueue_write_buffer(
                &mut temp_write_buffer,
                CL_NON_BLOCKING,
                0,
                &input_vectors_flattened,
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

        let outputs: Vec<Vec3h> = (0..n_inputs).into_iter()
            .map(|idx| Vec3h {
                x: output_data[idx * 4 + 0],
                y: output_data[idx * 4 + 1],
                z: output_data[idx * 4 + 2],
                w: output_data[idx * 4 + 3]
            }).collect::<Vec<Vec3h>>();

        Ok(outputs)
    }
}