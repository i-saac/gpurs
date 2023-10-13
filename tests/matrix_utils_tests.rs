use gpurs::Result;
use gpurs::linalg::{
    Matrix,
    MatrixUtilities,
    Axis
};

type P = f32;

#[test]
fn linear_solve_test() -> Result<()> {
    let a_mat: Matrix<P> = Matrix::new(vec![1.0, 1.0, 1.0, 3.0, 1.0, -3.0, 1.0, -2.0, -5.0], 3, 3)?;
    let b_vec: Matrix<P> = Matrix::new(vec![1.0, 5.0, 10.0], 3, 1)?;

    let x_vec: Matrix<P> = Matrix::linear_solve_matrix(&a_mat, &b_vec)?;

    println!("{}", x_vec);

    Ok(())
}

#[test]
fn max_min_test() -> Result<()> {
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
