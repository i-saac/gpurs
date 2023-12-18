mod test_parameters;
use test_parameters::P;

use gpurs::Result;
use gpurs::linalg::Matrix;

#[test]
fn indexing_tests() -> Result<()> {
    let mat_data: Vec<P> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut mat: Matrix<P> = Matrix::new(mat_data, 3, 4)?;

    assert_eq!(mat[[1, 2]], 7.0, "Index value not as expected");
    assert_eq!(mat.col_vec(2)?, vec![3.0, 7.0, 11.0], "Column vector not as expected");
    assert_eq!(mat.row_vec(0)?, vec![1.0, 2.0, 3.0, 4.0], "Row vector not as expected");
    assert_eq!(mat.slice_index(&[0, 2], &[2, 3]).get_data(), vec![3.0, 4.0, 11.0, 12.0], "Slice index not as expected");

    mat[[0, 3]] = 5.0;
    assert_eq!(mat.col_vec(3)?, vec![5.0, 8.0, 12.0], "Mutable index not working");

    Ok(())
}

#[test]
fn arithmetic_test() -> Result<()> {
    let a_vec: Vec<P> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_vec: Vec<P> = vec![2.0, 1.0, 2.0, 3.0, 2.0, 1.0];

    let a_mat: Matrix<P> = Matrix::new(a_vec, 2, 3)?;
    let b_mat: Matrix<P> = Matrix::new(b_vec, 2, 3)?;

    let b_transpose_vec: Vec<P> = vec![2.0, 3.0, 1.0, 2.0, 2.0, 1.0];
    let b_transpose: Matrix<P> = b_mat.clone().transpose();
    assert_eq!(b_transpose.get_data(), b_transpose_vec, "Transpose data not as expected");
    assert_eq!(b_transpose.get_rows(), 3, "Transpose row dimension not as expected");
    assert_eq!(b_transpose.get_cols(), 2, "Transpose col dimension not as expected");

    let value: P = 2.0;

    let fladd_vec: Vec<P> = vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let fladd_mat: Matrix<P> = a_mat.clone() + value;
    assert_eq!(fladd_mat.get_data(), fladd_vec, "Matrix-Float addition not as expected");

    let add_vec: Vec<P> = vec![3.0, 3.0, 5.0, 7.0, 7.0, 7.0];
    let add_mat: Matrix<P> = (a_mat.clone() + b_mat.clone())?;
    assert_eq!(add_mat.get_data(), add_vec, "Matrix-Matrix addition not as expected");

    let neg_vec: Vec<P> = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
    let neg_mat: Matrix<P> = -a_mat.clone();
    assert_eq!(neg_mat.get_data(), neg_vec, "Matrix negation not as expected");

    let flsub_vec: Vec<P> = vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    let flsub_mat: Matrix<P> = a_mat.clone() - value;
    assert_eq!(flsub_mat.get_data(), flsub_vec, "Matrix-Float subtraction not as expected");

    let sub_vec: Vec<P> = vec![-1.0, 1.0, 1.0, 1.0, 3.0, 5.0];
    let sub_mat: Matrix<P> = (a_mat.clone() - b_mat.clone())?;
    assert_eq!(sub_mat.get_data(), sub_vec, "Matrix-Matrix subtraction not as expected");

    let flmul_vec: Vec<P> = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
    let flmul_mat: Matrix<P> = value * a_mat.clone();
    assert_eq!(flmul_mat.get_data(), flmul_vec, "Matrix-Float multiplication not as expected");

    let mul_vec: Vec<P> = vec![10.0, 10.0, 25.0, 28.0];
    let mul_mat: Matrix<P> = (a_mat.clone() * b_transpose)?;
    assert_eq!(mul_mat.get_data(), mul_vec, "Matrix-Matrix multiplication data not as expected");
    assert_eq!(mul_mat.get_rows(), 2, "Matrix-Matrix multiplication row dimension not as expected");
    assert_eq!(mul_mat.get_cols(), 2, "Matrix-Matrix multiplication col dimension not as expected");

    let elementwise_vec: Vec<P> = vec![2.0, 2.0, 6.0, 12.0, 10.0, 6.0];
    let elementwise_mat: Matrix<P> = a_mat.elementwise_multiply(b_mat)?;
    assert_eq!(elementwise_mat.get_data(), elementwise_vec, "Elementwise multiplication data not as expected");
    assert_eq!(elementwise_mat.get_rows(), 2, "Elementwise multiplication row dimension not as expected");
    assert_eq!(elementwise_mat.get_cols(), 3, "Elementwise multiplication col dimension not as expected");

    Ok(())
}

#[test]
fn iter_test() -> Result<()> {
    let a_vec: Vec<P> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a_mat: Matrix<P> = Matrix::new(a_vec, 2, 3)?;

    for (idx, row) in a_mat.all_rows().enumerate() {
        println!("{}", row);
        assert_eq!(row.get_data(), a_mat.row_matrix(idx)?.get_data())
    }

    for (idx, col) in a_mat.all_cols().enumerate() {
        println!("{}", col);
        assert_eq!(col.get_data(), a_mat.col_matrix(idx)?.get_data())
    }

    for (idx, element) in a_mat.all_elements().enumerate() {
        println!("{}", element);
        assert_eq!(element, a_mat.lindex(idx))
    }

    Ok(())
}