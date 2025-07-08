//RELU.tpp
#pragma once

#include <algorithm> //std::max


template <typename Scalar>
MatrixD RELULayer<Scalar>::forward(const MatrixD& input){
    static_assert(std::is_arithmetic<Scalar>::value, "RELU forward: Scalar must be arithmetic.");
    assert(input.rows() > 0 && input.cols() > 0 && "RELU forward: input matrix must be non-empty.");
    
    this->input_cache = input;
    MatrixD output = input;
    for (int i=0; i < output.rows(); ++i){
        for(int j=0; j <output.cols(); ++j){
            output(i,j) = std::max(static_cast<Scalar>(0), input(i,j));
        }
    }
    return output;
}

template <typename Scalar>
MatrixD RELULayer<Scalar>::backward(const MatrixD& grad_output ){
    MatrixD grad_input = grad_output;
    for(int i=0; i< grad_input.rows(); ++i){
        for(int j=0; j <grad_input.cols(); ++j){
            grad_input(i,j) *= (this->input_cache(i,j) > 0 ? 1: 0);
        }
    }
    return grad_input;
}