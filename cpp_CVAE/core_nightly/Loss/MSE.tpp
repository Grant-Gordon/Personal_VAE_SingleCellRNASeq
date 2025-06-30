//MSE.tpp
#pragma once

template <typename MatrixType>
typename MSE<MatrixType>::Scalar MSE<MatrixType>::forward(const MatrixType& predicted, const MatrixType& target){
    static_assert(std::is_floating_point<typename MSE<MatrixType>::Scalar>::value,"MSE::forward: Scalar must be floating-point.");
    assert(predicted.rows() > 0
        && "MSE::forward: predicted must have non-zero rows.");
    assert(predicted.rows() == target.rows() 
        && predicted.cols() == target.cols()
        && "MSE::forward: predicted and target must have the same shape.");

    
    MatrixType diff = predicted - target;
    return diff.squaredNorm() / static_cast<typename MatrixType::Scalar>(predicted.rows());
}

template <typename MatrixType>
MatrixType MSE<MatrixType>::backward(const MatrixType& predicted, const MatrixType& target){
    
    assert(predicted.rows() > 0 
        && "MSE::backward: predicted must have non-zero rows.");
    assert(predicted.rows() == target.rows() 
        && predicted.cols() == target.cols() 
        && "MSE::backward: predicted and target must have the same shape.");

    return (predicted - target) * (2.0 / static_cast<typename MatrixType::Scalar>(predicted.rows()));
}