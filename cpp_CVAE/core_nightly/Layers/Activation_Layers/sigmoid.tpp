//TODO: needs like an 90% refactor

#pragma once
#include <cmath>

//forward
template <typename MatrixType>
MatrixType SigmoidLayer<MatrixType>::forward(const MatrixType& input){
    MatrixType output = input;
    for (int i = 0; i < input.rows(); ++i){
        for (int j = 0; j < input.cols() ++j){
            output(i,j) = static_cast<Scalar>(1) / (static_cast<Scalar>(1) + std::exp(-input(i,j)));
        }
    }
    this->output_cache = output;
    return output;
}

template <typename MatrixType>
MatrixType SigmoidLayer<MatrixType>::backward(const MatrixType& grad_output){
    MatrixType grad_input = grad_output;
    for(int i=0; i < grad_output.rows(); ++i){
        for(int j=0; j< grad_output.cols(); ++j){
            Scalar sigmoid_output = this->output_cache(i,j);
            grad_input(i,j) *= sigmoid_output * (static_const<Scalar>(1) - sigmoid_output); //dL/dx = dL/dy * dy/dx , dL/dy = grad_ouput, derivitive sigmoid = sig(x) * (1-sig(x))
        }
    }
    return grad_input;
}