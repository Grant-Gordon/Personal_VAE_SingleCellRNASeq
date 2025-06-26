#pragma once

template <typename MatrixType>
typename MSE<MatrixType>::Scalar MSE<MatrixType>::forward(const MatrixType& predicted, const MatrixType& target){
    MatrixType diff = predicted - target;
    return diff.squaredNorm() / static_cast<typename MatrixType::Scalar>(predicted.rows());
}

template <typename MatrixType>
MatrixType MSE<MatrixType>::backward(const MatrixType& predicted, const MatrixType& target){
    return (predicted - target) * (2.0 / static_cast<typename MatrixType::Scalar>(predicted.rows()));
}