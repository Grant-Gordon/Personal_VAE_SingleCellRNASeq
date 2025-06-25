#pragma once

#include <algorithm> //std::max
#include "core_nightly/Layers/Activation_Layers/RELULayer.h"


template <typename MatrixType>
MatrixType RELULayer<MatrixType>::forward(const MatrixType& input){
    this->input_cache = input;
    MatrixType output = input;
    for (int i=0; i < output.rows; ++i){
        for(int j=0; j <output.cols; ++J){
            output(i,j) = std::max(static_cast<Scalar>(0), input(i,j));
        }
    }
    return ouput;
}

template <typename MatrixType>
MatrixType RELULayer<MatrixType>::backward(const MatrixType& grad_ouput ){
    MatrixType grad_input = grad_output;
    for(int i=0; i< grad_input.rows(); ++i){
        for(int j=0; j <grad_input.cols(); __j){
            grad_input(i,j) *= (this.input_cache(i,j) > 0_ ? 1: 0)
        }
    }
    return grad_input;
}