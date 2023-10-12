#![cfg(feature = "gpu_accel")]

use std::time::Instant;

use gpurs::Result;
use gpurs::Jeeperr;
use gpurs::linalg::Matrix;
use gpurs::gpu::{
    MemoryCalculator,
    MemoryParameterFunction
};

#[test]
fn memory_gpu_matmul_test() {
    let mut calc: MemoryCalculator<f64> = MemoryCalculator::<f64>::init()
        .expect("Failed to initialize calculator");

    let a_vec: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_vec: Vec<f64> = vec![2.0, 1.0, 2.0, 3.0, 2.0, 1.0];

    let c_vec: Vec<f64> = vec![12.0, 10.0, 30.0, 25.0];
    let d_vec: Vec<f64> = vec![52.0, 74.0, 96.0, 130.0, 185.0, 240.0];

    let e_vec: Vec<f64> = vec![6.0, 9.0, 12.0, 14.0, 19.0, 24.0, 6.0, 9.0, 12.0];
    let f_vec: Vec<f64> = vec![54.0, 45.0, 114.0, 95.0, 54.0, 45.0];

    let a_mat: Matrix<f64> = Matrix::new(a_vec, 2, 3)
        .expect("Failed to create Matrix A");
    let b_mat: Matrix<f64> = Matrix::new(b_vec, 3, 2)
        .expect("Failed to create Matrix B");

    let a_idx: usize = calc.store_matrix(a_mat)
        .expect("Failed to store Matrix A in calculator memory");
    let b_idx: usize = calc.store_matrix(b_mat)
        .expect("Failed to store Matrix B in calculator memory");

    let (c_mat, c_idx) = calc.mat_mul(a_idx, b_idx)
        .expect("Failed to mulitply Matrix A and Matrix B");

    assert_eq!(c_mat.get_data(), c_vec, "Matrix C data not as expected");
    assert_eq!(c_mat.get_rows(), 2, "Matrix C row dimension not as expected");
    assert_eq!(c_mat.get_cols(), 2, "Matrix C col dimension not as expected");

    let (d_mat, _) = calc.mat_mul(c_idx, a_idx)
        .expect("Failed to multiply Matrix C and Matrix A");

    assert_eq!(d_mat.get_data(), d_vec, "Matrix D data not as expected");
    assert_eq!(d_mat.get_rows(), 2, "Matrix D row dimension not as expected");
    assert_eq!(d_mat.get_cols(), 3, "Matrix D col dimension not as expected");

    let (e_mat, e_idx) = calc.mat_mul(b_idx, a_idx)
        .expect("Failed to mulitply Matrix B and Matrix A");

    assert_eq!(e_mat.get_data(), e_vec, "Matrix E data not as expected");
    assert_eq!(e_mat.get_rows(), 3, "Matrix E row dimension not as expected");
    assert_eq!(e_mat.get_cols(), 3, "Matrix E col dimension not as expected");

    let (f_mat, _) = calc.mat_mul(e_idx, b_idx)
        .expect("Failed to multiply Matrix E and Matrix B");

    assert_eq!(f_mat.get_data(), f_vec, "Matrix F data not as expected");
    assert_eq!(f_mat.get_rows(), 3, "Matrix F row dimension not as expected");
    assert_eq!(f_mat.get_cols(), 2, "Matrix F col dimension not as expected");
}

#[test]
fn memory_custom_kernel_test() {
    let mut calc: MemoryCalculator<f64> = MemoryCalculator::<f64>::init()
        .expect("Failed to initialize calculator");

    let new_program: &str = r#"
    kernel void mat_ewmult (
        global double *c,
        const int N,
        const global double* a,
        const global double* b
    ) {
        const int globalRow = get_global_id(0);
        const int globalCol = get_global_id(1);
    
        c[globalRow * N + globalCol] = a[globalRow * N + globalCol] * b[globalRow * N + globalCol];
    }
    "#;

    let new_kernel_name: &str = "mat_ewmult";

    let custom_param_function: MemoryParameterFunction<f64> = Box::new(
        | input_mats: Vec<&Matrix<f64>> | -> Result<(usize, usize, Vec<usize>)> {
            if input_mats.len() != 2 {
                return Err(Jeeperr::ArgumentError)
            }

            let left: &Matrix<f64> = input_mats[0];
            let right: &Matrix<f64> = input_mats[1];

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
        calc.load_custom_fn(new_program, new_kernel_name, custom_param_function)
            .expect("Failed to load custom function")
    };

    let a_vec: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_vec: Vec<f64> = vec![2.0, 1.0, 2.0, 3.0, 2.0, 1.0];
    
    let c_vec: Vec<f64> = vec![2.0, 2.0, 6.0, 12.0, 10.0, 6.0];

    let a_mat: Matrix<f64> = Matrix::new(a_vec, 2, 3)
        .expect("Failed to create Matrix A");
    let b_mat: Matrix<f64> = Matrix::new(b_vec, 2, 3)
        .expect("Failed to create Matrix B");

    let a_idx: usize = calc.store_matrix(a_mat)
        .expect("Failed to store Matrix A in calculator memory");
    let b_idx: usize = calc.store_matrix(b_mat)
        .expect("Failed to store Matrix B in calculator memory");

    let (c_mat, _) = unsafe {
        calc.exec_custom_fn(
            custom_idx,
            None,
            Some(vec![3 as i32]),
            vec![a_idx, b_idx],
        ).expect("Failed to multiply Matrix A and Matrix B elementwise")
    };

    assert_eq!(c_mat.get_data(), c_vec, "Matrix C data not as expected");
    assert_eq!(c_mat.get_rows(), 2, "Matrix C row dimension not as expected");
    assert_eq!(c_mat.get_cols(), 3, "Matrix C row dimension not as expected");
}

#[test]
fn cpu_vs_memory_gpu_test() {
    let a_mat: Matrix<f64> = Matrix::<f64>::ones(1000, 1000);
    let b_mat: Matrix<f64> = Matrix::<f64>::ones(1000, 1000);
    let c_mat: Matrix<f64> = Matrix::<f64>::ones(1000, 1000);

    let gpu_total_start = Instant::now();
    let output_gpu: Matrix<f64>;
    {
        let mut calc: MemoryCalculator<f64> = MemoryCalculator::<f64>::init()
            .expect("Failed to initialize calculator");

        let a_idx: usize = calc.store_matrix(a_mat)
            .expect("Failed to store Matrix A");

        let gpu_calc_start = Instant::now();
        (output_gpu, _) = calc.mat_mul(a_idx, a_idx)
            .expect("Failed to multiply Matrix A by Matrix A (GPU)");
        println!("GPU Calc: {:?}", gpu_calc_start.elapsed());
    }
    println!("GPU Total: {:?}", gpu_total_start.elapsed());

    let cpu_start = Instant::now();
    let output_cpu = (b_mat * c_mat).expect("failed to multiply Matrix A by Matrix A (CPU)");
    println!("CPU: {:?}", cpu_start.elapsed());

    assert_eq!(output_gpu.get_data(), output_cpu.get_data())
}