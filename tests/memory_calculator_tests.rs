#![cfg(feature = "gpu_accel")]

use std::time::Instant;

use gpurs::Result;
use gpurs::Jeeperr;
use gpurs::linalg::Matrix;
use gpurs::gpu::{
    MemoryCalculator,
    MemoryParameterFunction
};

type P = f32;

#[test]
fn memory_gpu_matmul_test() -> Result<()> {
    let mut calc: MemoryCalculator<P> = MemoryCalculator::<P>::init()?;

    let a_vec: Vec<P> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_vec: Vec<P> = vec![2.0, 1.0, 2.0, 3.0, 2.0, 1.0];

    let c_vec: Vec<P> = vec![12.0, 10.0, 30.0, 25.0];
    let d_vec: Vec<P> = vec![52.0, 74.0, 96.0, 130.0, 185.0, 240.0];

    let e_vec: Vec<P> = vec![6.0, 9.0, 12.0, 14.0, 19.0, 24.0, 6.0, 9.0, 12.0];
    let f_vec: Vec<P> = vec![54.0, 45.0, 114.0, 95.0, 54.0, 45.0];

    let a_mat: Matrix<P> = Matrix::new(a_vec, 2, 3)?;
    let b_mat: Matrix<P> = Matrix::new(b_vec, 3, 2)?;

    let a_idx: usize = calc.store_matrix(a_mat)?;
    let b_idx: usize = calc.store_matrix(b_mat)?;

    let (c_mat, c_idx) = calc.mat_mul(a_idx, b_idx)?;

    assert_eq!(c_mat.get_data(), c_vec, "Matrix C data not as expected");
    assert_eq!(c_mat.get_rows(), 2, "Matrix C row dimension not as expected");
    assert_eq!(c_mat.get_cols(), 2, "Matrix C col dimension not as expected");

    let (d_mat, _) = calc.mat_mul(c_idx, a_idx)?;

    assert_eq!(d_mat.get_data(), d_vec, "Matrix D data not as expected");
    assert_eq!(d_mat.get_rows(), 2, "Matrix D row dimension not as expected");
    assert_eq!(d_mat.get_cols(), 3, "Matrix D col dimension not as expected");

    let (e_mat, e_idx) = calc.mat_mul(b_idx, a_idx)?;

    assert_eq!(e_mat.get_data(), e_vec, "Matrix E data not as expected");
    assert_eq!(e_mat.get_rows(), 3, "Matrix E row dimension not as expected");
    assert_eq!(e_mat.get_cols(), 3, "Matrix E col dimension not as expected");

    let (f_mat, _) = calc.mat_mul(e_idx, b_idx)?;

    assert_eq!(f_mat.get_data(), f_vec, "Matrix F data not as expected");
    assert_eq!(f_mat.get_rows(), 3, "Matrix F row dimension not as expected");
    assert_eq!(f_mat.get_cols(), 2, "Matrix F col dimension not as expected");

    Ok(())
}

#[test]
fn memory_custom_kernel_test() -> Result<()> {
    let mut calc: MemoryCalculator<P> = MemoryCalculator::<P>::init()?;

    let new_program: &str = r#"
    kernel void mat_ewmult (
        global float *c,
        const int N,
        const global float* a,
        const global float* b
    ) {
        const int globalRow = get_global_id(0);
        const int globalCol = get_global_id(1);
    
        c[globalRow * N + globalCol] = a[globalRow * N + globalCol] * b[globalRow * N + globalCol];
    }
    "#;

    let new_kernel_name: &str = "mat_ewmult";

    let custom_param_function: MemoryParameterFunction<P> = Box::new(
        | input_mats: Vec<&Matrix<P>> | -> Result<(usize, usize, Vec<usize>)> {
            if input_mats.len() != 2 {
                return Err(Jeeperr::ArgumentError)
            }

            let left: &Matrix<P> = input_mats[0];
            let right: &Matrix<P> = input_mats[1];

            if (left.get_rows() != right.get_rows())
                || (left.get_cols() != right.get_cols())
            {
                return Err(Jeeperr::DimensionError)
            }

            let output_rows: usize = left.get_rows();
            let output_cols: usize = left.get_cols();
            let work_sizes: Vec<usize> = vec![left.get_rows(), left.get_cols()];

            Ok((output_rows, output_cols, work_sizes))
        }
    );

    let custom_idx: usize = unsafe {
        calc.load_custom_fn(new_program, new_kernel_name, custom_param_function)?
    };

    let a_vec: Vec<P> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_vec: Vec<P> = vec![2.0, 1.0, 2.0, 3.0, 2.0, 1.0];
    
    let c_vec: Vec<P> = vec![2.0, 2.0, 6.0, 12.0, 10.0, 6.0];

    let a_mat: Matrix<P> = Matrix::new(a_vec, 2, 3)?;
    let b_mat: Matrix<P> = Matrix::new(b_vec, 2, 3)?;

    let a_idx: usize = calc.store_matrix(a_mat)?;
    let b_idx: usize = calc.store_matrix(b_mat)?;

    let (c_mat, _) = unsafe {
        calc.exec_custom_fn(
            custom_idx,
            None,
            Some(vec![3 as i32]),
            vec![a_idx, b_idx],
        )?
    };

    assert_eq!(c_mat.get_data(), c_vec, "Matrix C data not as expected");
    assert_eq!(c_mat.get_rows(), 2, "Matrix C row dimension not as expected");
    assert_eq!(c_mat.get_cols(), 3, "Matrix C row dimension not as expected");

    Ok(())
}

#[test]
fn cpu_vs_memory_gpu_test() -> Result<()> {
    let a_mat: Matrix<P> = Matrix::<P>::ones(1000, 1000);
    let b_mat: Matrix<P> = Matrix::<P>::ones(1000, 1000);
    let c_mat: Matrix<P> = Matrix::<P>::ones(1000, 1000);

    let gpu_total_start = Instant::now();
    let output_gpu: Matrix<P>;
    {
        let mut calc: MemoryCalculator<P> = MemoryCalculator::<P>::init()?;

        let a_idx: usize = calc.store_matrix(a_mat)?;

        let gpu_calc_start = Instant::now();
        (output_gpu, _) = calc.mat_mul(a_idx, a_idx)?;
        println!("GPU Calc: {:?}", gpu_calc_start.elapsed());
    }
    println!("GPU Total: {:?}", gpu_total_start.elapsed());

    let cpu_start = Instant::now();
    let output_cpu = (b_mat * c_mat)?;
    println!("CPU: {:?}", cpu_start.elapsed());

    assert_eq!(output_gpu.get_data(), output_cpu.get_data());

    Ok(())
}