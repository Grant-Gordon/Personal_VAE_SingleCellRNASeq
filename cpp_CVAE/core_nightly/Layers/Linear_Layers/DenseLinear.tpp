//DenseLinear.tpp
#pragma once
#include <omp.h>
#include <random>
#include <config_values.h>
#include "custom_types.h"
#include "param_init_utils.h"
#include "macros.h"


template <typename Scalar>
DenseLinear<Scalar>::DenseLinear(
    unsigned int input_dim,
    unsigned int output_dim,
    InitFn<Scalar> init_fn
)
{
    this->input_dim = input_dim;
    this->output_dim = output_dim;
    std::mt19937 gen(configV::Global__seed);
    this->weights = MatrixD<Scalar>(output_dim, input_dim);
    ASSERT(input_dim > 0 && output_dim > 0);
    this-> bias = VectorD<Scalar>::Zero(output_dim);
    
    for (size_t i = 0; i < output_dim; ++i){
        this->bias(0,i) = init_fn(input_dim, output_dim, gen); //TODO: confirm order of args is correct
        ASSERT(std::isfinite(this->bias(0,i)));
        for (size_t j = 0; j < input_dim; ++j){
            this->weights(i,j) = init_fn(input_dim, output_dim, gen);
            ASSERT(std::isfinite(this->weights(i,j)));
        }
    }
}


template <typename Scalar>
MatrixD<Scalar> DenseLinear<Scalar>::forward(const MatrixD<Scalar>& input){
    this->input_cache = input;
    ASSERT(input.cols() == this->weights.cols());
    DASSERT(this->bias.size() == this->weights.rows());
    //y = xW^T + b (broadcasted): where input = (batch_size X features), W = (inputFeature X outputFeature), bias = (1 X output_size), input*W = (batch_size X output_features)
    return (input * this->weights.transpose()).rowwise() + this->bias.transpose();
}

template <typename Scalar>
MatrixD<Scalar> DenseLinear<Scalar>::backward(const MatrixD<Scalar>& upstream_grad){
    
    
    //  = dL/dW 
    //  = dL/dy *dy/dW
    //      dL/dy = upstream_grad
    //      dy/dW = x | specifically if y_i = [sum over j (w_ij * x_i + b_i)]  then dy_i/dW_ij = x_j
    //  = upstream_grad^T * input
    this->grad_weights = upstream_grad.transpose() * this->input_cache;
    ASSERT(upstream_grad.cols() == this->weights.rows());
    ASSERT(this->input_cache.cols() == this->weights.cols());
    DASSERT(this->grad_weights.rows() == this->weights.rows());
    DASSERT(this->grad_weights.cols() == this->weights.cols());
    //grad_bias [out_d *1]
    //  = dL/db 
    //  = dL/dy * dy/db
    //      dL/dy = upstream_grad
    //      dy/db = I | specifically if y_i = W_i * x + b_i, then dy/db_i = 1, for all i!=j, dy/db_i =0 i.e. I
    //  = upstream_grad * Identity
    //  = sum across rows
    this->grad_bias = upstream_grad.colwise().sum().transpose();
    DASSERT(this->grad_bias.size() == this->bias.size());
    

    //grad_input [B * in_d]
    //  = dL/dx 
    //  = dL/dy * dy/dx 
    //      dL/dy = upstream_grad
    //      dy/dx = W 
    //  = upstream_grad * Weights
    DASSERT((upstream_grad * this->weights).cols() == this->input_cache.cols());
    return upstream_grad * this->weights;
}

template <typename Scalar>
const MatrixD<Scalar>& DenseLinear<Scalar>::get_input_cache()const{
    return this->input_cache;
}

template <typename Scalar>
MatrixD<Scalar>& DenseLinear<Scalar>::get_input_cache(){
    return this->input_cache;
}