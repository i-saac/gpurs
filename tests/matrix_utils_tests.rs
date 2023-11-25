mod test_parameters;
use test_parameters::P;

use gpurs::Result;
use gpurs::linalg::{
    Matrix,
    MatrixUtilities,
    Axis,
    BooleanMatrixOperations,
    ReferenceOperations
};

#[test]
fn linear_solve_test() -> Result<()> {
    let a_mat: Matrix<P> = Matrix::new(vec![1.0, 1.0, 1.0, 3.0, 1.0, -3.0, 1.0, -2.0, -5.0], 3, 3)?;
    let b_vec: Matrix<P> = Matrix::new(vec![1.0, 5.0, 10.0], 3, 1)?;

    let x_vec: Matrix<P> = Matrix::linear_solve_matrix(&a_mat, &b_vec)?;

    println!("{}", x_vec);

    Ok(())
}

#[test]
fn max_min_sum_test() -> Result<()> {
    let a_data: Vec<P> = vec![1.0, 2.0, 3.0, 4.0];
    let a_mat: Matrix<P> = Matrix::new(a_data, 2, 2)?;

    let max_val: P = Matrix::max(&a_mat);
    assert_eq!(max_val, 4.0, "Max value not as expected");

    let row_max: Matrix<P> = Matrix::axis_max(&a_mat, Axis::Row);
    assert_eq!(row_max.get_data(), &[2.0, 4.0], "Max values along row axis not as expected");

    let col_max: Matrix<P> = Matrix::axis_max(&a_mat, Axis::Col);
    assert_eq!(col_max.get_data(), &[3.0, 4.0], "Max values along col axis not as expected");

    let min_val: P = Matrix::min(&a_mat);
    assert_eq!(min_val, 1.0, "Min value not as expected");

    let row_min: Matrix<P> = Matrix::axis_min(&a_mat, Axis::Row);
    assert_eq!(row_min.get_data(), &[1.0, 3.0], "Min values along row axis not as expected");

    let col_min: Matrix<P> = Matrix::axis_min(&a_mat, Axis::Col);
    assert_eq!(col_min.get_data(), &[1.0, 2.0], "Min values along col axis not as expected");

    let sum_val: P = Matrix::sum(&a_mat);
    assert_eq!(sum_val, 10.0, "Sum value not as expected");

    let row_sum: Matrix<P> = Matrix::axis_sum(&a_mat, Axis::Row);
    assert_eq!(row_sum.get_data(), &[3.0, 7.0], "Sum values along row axis not as expected");

    let col_sum: Matrix<P> = Matrix::axis_sum(&a_mat, Axis::Col);
    assert_eq!(col_sum.get_data(), &[4.0, 6.0], "Sum values along col axis not as expected");

    Ok(())
}

#[test]
fn concatenation_test() -> Result<()> {
    let a_data: Vec<P> = vec![1.0, 2.0, 3.0, 4.0];
    let a_mat: Matrix<P> = Matrix::new(a_data, 2, 2)?;

    println!("{}", a_mat);

    let row_concat: Matrix<P> = Matrix::concatenate(&[a_mat.clone(), a_mat.clone(), a_mat.clone()], Axis::Row)?;
    assert_eq!(row_concat.get_data(), &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], "Row concatenation data not as expected");
    assert_eq!(row_concat.get_rows(), 6, "Row concatenation row dimension not as expected");
    assert_eq!(row_concat.get_cols(), 2, "Row concatenation col dimension not as expected");

    println!("{}", row_concat);

    let col_concat: Matrix<P> = Matrix::concatenate(&[a_mat.clone(), a_mat.clone(), a_mat.clone()], Axis::Col)?;
    assert_eq!(col_concat.get_data(), &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0], "Col concatenation data not as expected");
    assert_eq!(col_concat.get_rows(), 2, "Col concatenation row dimension not as expected");
    assert_eq!(col_concat.get_cols(), 6, "Col concatenation col dimension not as expected");

    println!("{}", col_concat);

    Ok(())
}

#[test]
fn bool_ops() -> Result<()> {
    let a_vec: Vec<P> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a_mat: Matrix<P> = Matrix::new(a_vec, 2, 3)?;

    let equal_vec: Vec<bool> = Matrix::equal(&a_mat, 3.0);
    assert_eq!(equal_vec, vec![false, false, true, false, false, false], "Equal to vector not as expected");

    let less_than_vec: Vec<bool> = Matrix::less_than(&a_mat, 3.0);
    assert_eq!(less_than_vec, vec![true, true, false, false, false, false], "Less than vector not as expected");
    let less_equal_vec: Vec<bool> = Matrix::less_equal(&a_mat, 3.0);
    assert_eq!(less_equal_vec, vec![true, true, true, false, false, false], "Less equal vector not as expected");

    let greater_than_vec: Vec<bool> = Matrix::greater_than(&a_mat, 3.0);
    assert_eq!(greater_than_vec, vec![false, false, false, true, true, true], "Greater than vector not as expected");
    let greater_equal_vec: Vec<bool> = Matrix::greater_equal(&a_mat, 3.0);
    assert_eq!(greater_equal_vec, vec![false, false, true, true, true, true], "Greater equal vector not as expected");

    let b_vec: Vec<P> = vec![2.0, 2.0, 4.0, 3.0, 2.0, 1.0];
    let b_mat: Matrix<P> = Matrix::new(b_vec, 2, 3)?;

    let equal_mat_vec: Vec<bool> = Matrix::equal_mat(&a_mat, &b_mat)?;
    assert_eq!(equal_mat_vec, vec![false, true, false, false, false, false], "Equal to mat vector not as expected");
    
    let less_than_mat_vec: Vec<bool> = Matrix::less_than_mat(&a_mat, &b_mat)?;
    assert_eq!(less_than_mat_vec, vec![true, false, true, false, false, false], "Less than mat vector not as expected");
    let less_equal_mat_vec: Vec<bool> = Matrix::less_equal_mat(&a_mat, &b_mat)?;
    assert_eq!(less_equal_mat_vec, vec![true, true, true, false, false, false], "Less equal mat vector not as expected");

    let greater_than_mat_vec: Vec<bool> = Matrix::greater_than_mat(&a_mat, &b_mat)?;
    assert_eq!(greater_than_mat_vec, vec![false, false, false, true, true, true], "Greater than mat vector not as expected");
    let greater_equal_mat_vec: Vec<bool> = Matrix::greater_equal_mat(&a_mat, &b_mat)?;
    assert_eq!(greater_equal_mat_vec, vec![false, true, false, true, true, true], "Greater equal mat vector not as expected");

    Ok(())
}

#[test]
fn ref_ops() -> Result<()> {
    let a_vec: Vec<P> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_vec: Vec<P> = vec![2.0, 1.0, 2.0, 3.0, 2.0, 1.0];

    let a_mat: Matrix<P> = Matrix::new(a_vec, 2, 3)?;
    let b_mat: Matrix<P> = Matrix::new(b_vec, 2, 3)?;

    let add_vec: Vec<P> = vec![3.0, 3.0, 5.0, 7.0, 7.0, 7.0];
    let add_mat: Matrix<P> = Matrix::ref_add(&a_mat, &b_mat)?;
    assert_eq!(add_mat.get_data(), add_vec, "Matrix-Matrix addition not as expected");

    let neg_vec: Vec<P> = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
    let neg_mat: Matrix<P> = Matrix::ref_neg(&a_mat);
    assert_eq!(neg_mat.get_data(), neg_vec, "Matrix negation not as expected");

    let sub_vec: Vec<P> = vec![-1.0, 1.0, 1.0, 1.0, 3.0, 5.0];
    let sub_mat: Matrix<P> = Matrix::ref_sub(&a_mat, &b_mat)?;
    assert_eq!(sub_mat.get_data(), sub_vec, "Matrix-Matrix subtraction not as expected");

    let mul_vec: Vec<P> = vec![10.0, 10.0, 25.0, 28.0];
    let mul_mat: Matrix<P> = Matrix::ref_mul(&a_mat, &b_mat.transpose())?;
    assert_eq!(mul_mat.get_data(), mul_vec, "Matrix-Matrix multiplication data not as expected");
    assert_eq!(mul_mat.get_rows(), 2, "Matrix-Matrix multiplication row dimension not as expected");
    assert_eq!(mul_mat.get_cols(), 2, "Matrix-Matrix multiplication col dimension not as expected");

    let elementwise_vec: Vec<P> = vec![2.0, 2.0, 6.0, 12.0, 10.0, 6.0];
    let elementwise_mat: Matrix<P> = Matrix::ref_ewm(&a_mat, &b_mat)?;
    assert_eq!(elementwise_mat.get_data(), elementwise_vec, "Elementwise multiplication data not as expected");
    assert_eq!(elementwise_mat.get_rows(), 2, "Elementwise multiplication row dimension not as expected");
    assert_eq!(elementwise_mat.get_cols(), 3, "Elementwise multiplication col dimension not as expected");

    Ok(())
}