#![cfg(feature = "gpu_accel")]

use gpurs::Result;
use gpurs::Jeeperr;
use gpurs::linalg::Matrix;
use gpurs::gpu::{
    QuickCalculator,
    QuickParameterFunction
};

type P = f32;

#[test]
fn memory_gpu_matmul_test() -> Result<()> {
    let mut calc: QuickCalculator<P> = QuickCalculator::<P>::init()?;

    let a_vec: Vec<P> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_vec: Vec<P> = vec![2.0, 1.0, 2.0, 3.0, 2.0, 1.0];

    let c_vec: Vec<P> = vec![12.0, 10.0, 30.0, 25.0];
    let d_vec: Vec<P> = vec![54.0, 45.0, 114.0, 95.0, 54.0, 45.0];

    let a_mat: Matrix<P> = Matrix::new(a_vec, 2, 3)?;
    let b_mat: Matrix<P> = Matrix::new(b_vec, 3, 2)?;

    let c_mat: Matrix<P> = calc.quick_mat_mul(&a_mat, &b_mat)?;

    assert_eq!(c_mat.get_data(), c_vec, "Matrix C data not as expected");
    assert_eq!(c_mat.get_rows(), 2, "Matrix C row dimension not as expected");
    assert_eq!(c_mat.get_cols(), 2, "Matrix C col dimension not as expected");

    let b_idx: usize = calc.store_matrix(b_mat)?;

    let d_mat: Matrix<P> = calc.halfquick_mat_mul(b_idx, &c_mat)?;

    assert_eq!(d_mat.get_data(), d_vec, "Matrix D data not as expected");
    assert_eq!(d_mat.get_rows(), 3, "Matrix D row dimension not as expected");
    assert_eq!(d_mat.get_cols(), 2, "Matrix D col dimension not as expected");

    Ok(())
}

#[test]
fn quick_custom_kernel_test() -> Result<()> {
    let mut calc: QuickCalculator<P> = QuickCalculator::<P>::init()?;

    let new_program: &str = r#"
    kernel void ax_plus_b (
        global float* c,
        const int N,
        const global float* a,
        const global float* b,
        const global float* x
    ) {
        const int globalOutputIdx = get_global_id(0);
    
        float interm = 0.0f;
        for (int k = 0; k < N; k++) {
            interm += a[globalOutputIdx * N + k] * x[k];
        }
    
        c[globalOutputIdx] = interm + b[globalOutputIdx];
    }
    "#;

    let new_kernel_name: &str = "ax_plus_b";

    let custom_param_function: QuickParameterFunction<P> = Box::new(
        | input_stored_mats: Option<Vec<&Matrix<P>>>, input_temp_mats: Option<Vec<&Matrix<P>>> |
            -> Result<(usize, usize, Vec<usize>)>
        {
            if input_stored_mats.is_none() || input_temp_mats.is_none() {
                return Err(Jeeperr::ArgumentError)
            }

            let stored_mats: Vec<&Matrix<P>> = input_stored_mats.unwrap();
            let temp_mats: Vec<&Matrix<P>> = input_temp_mats.unwrap();
            
            if stored_mats.len() != 2 {
                return Err(Jeeperr::ArgumentError)
            }
            if temp_mats.len() != 1 {
                return Err(Jeeperr::ArgumentError)
            }

            let a: &Matrix<P> = stored_mats[0];
            let b: &Matrix<P> = stored_mats[1];

            let x: &Matrix<P> = temp_mats[0];

            if a.get_cols() != x.get_rows() {
                return Err(Jeeperr::DimensionError)
            }
            if x.get_cols() != 1 || b.get_cols() != 1 {
                return Err(Jeeperr::DimensionError)
            }
            if a.get_rows() != b.get_rows() {
                return Err(Jeeperr::DimensionError)
            }

            let output_rows: usize = a.get_rows();
            let output_cols: usize = 1;
            let work_sizes: Vec<usize> = vec![a.get_rows()];

            Ok((output_rows, output_cols, work_sizes))
        }
    );

    let custom_idx: usize = unsafe {
        calc.load_custom_fn(new_program, new_kernel_name, custom_param_function)?
    };

    let a_vec: Vec<P> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_vec: Vec<P> = vec![2.0, 5.0];
    
    let x_vec: Vec<P> = vec![1.0, 0.0, 2.0];

    let c_vec: Vec<P> = vec![9.0, 21.0];

    let a_mat: Matrix<P> = Matrix::new(a_vec, 2, 3)?;
    let b_mat: Matrix<P> = Matrix::new(b_vec, 2, 1)?;

    let x_mat: Matrix<P> = Matrix::new(x_vec, 3, 1)?;

    let a_idx: usize = calc.store_matrix(a_mat)?;
    let b_idx: usize = calc.store_matrix(b_mat)?;

    let c_mat = unsafe {
        calc.exec_custom_fn(
            custom_idx,
            None,
            Some(vec![3 as i32]),
            Some(vec![a_idx, b_idx]),
            Some(vec![&x_mat])
        )?
    };

    assert_eq!(c_mat.get_data(), c_vec, "Matrix C data not as expected");
    assert_eq!(c_mat.get_rows(), 2, "Matrix C row dimension not as expected");
    assert_eq!(c_mat.get_cols(), 1, "Matrix C row dimension not as expected");

    Ok(())
}