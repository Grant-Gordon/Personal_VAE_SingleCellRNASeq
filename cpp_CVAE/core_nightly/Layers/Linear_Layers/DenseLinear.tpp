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
    VERBOSEL2("Inside DenseLinear Constructor");
    ASSERT(input_dim > 0 && output_dim > 0); //Cannot have negative inputs
   
    this->input_dim = input_dim;
    this->output_dim = output_dim;
    std::mt19937 gen(configV::Global__seed);
    
   
    this->weights = MatrixD<Scalar>(output_dim, input_dim);
    this-> bias = VectorD<Scalar>::Zero(output_dim);
   
    //Validate W/b shapes
    DASSERT(this->weights.rows() == this->output_dim && this->weights.cols() == this->input_dim);
    DASSERT(this->bias.rows() == this->output_dim && this->bias.cols() == 1);
    
    for (size_t i = 0; i < output_dim; ++i){
        this->bias(i) = init_fn(input_dim, output_dim, gen);
       
        DASSERT(std::isfinite(this->bias(i)));
       
        for (size_t j = 0; j < input_dim; ++j){
            this->weights(i,j) = init_fn(input_dim, output_dim, gen);
           
            DASSERT(std::isfinite(this->weights(i,j)));
        }
    }
    VERBOSEL2("Finished Constructing DenseLinear object");
}

template <typename Scalar>
MatrixD<Scalar> DenseLinear<Scalar>::forward(const MatrixD<Scalar>& input){ //TODO could enforece passing by R-val to avoid copying at assignment of input_cache, but supposedly Eigens move semnatics are not neccissarily faster????
    VERBOSEL2("Inside DenseLinear::forward"); 
/*
SHAPE ASSERTIONS: 
    input:         [batch_size × input_dim]
    input_cache:   [batch_size × input_dim]
    weights:       [output_dim × input_dim]
    bias:          [output_dim × 1]
    weights^T:     [input_dim × output_dim]
    output:        [batch_size × output_dim]
*/
    //Validate input shape
    ASSERT(input.cols() == this->input_size);
    ASSERT(input.rows() <= configV::Training__batch_size && input.rows() > 0);    
    //Confirm W/b shape
    DASSERT(this->weights.rows() == this->output_dim && this->weights.cols() == this->input_dim);
    DASSERT(this->bias.rows() == this->output_dim && this->bias.cols() == 1);
    
    this->input_cache = input;
    
    //Check for successful Assignment
    DASSERT(input_cache.cols() == this->input_size);
    DASSERT(input_cache.rows() <= configV::Training__batch_size && input_cache.rows() > 0);    

    //y = xW^T + b (broadcasted): where input = [batch_size X input_dim], W = [output_dim X input_dim], bias = [output_dim X 1], input*W^T = [batch_size X input_dim] * [input_dim X output_dim] = [batch_sizd X output_dim]
    MatrixD<Scalar> output = (input * this->weights.transpose()).rowwise() + this->bias.transpose();
   
    //Validate Output Dimensions
    DASSERT(output.rows() == input.rows()); // == batchsize
    DASSERT(output.cols() == this->output_dim);

    VERBOSEL2("Finished with DenseLinear::forward");
    return output;
}

template <typename Scalar>
MatrixD<Scalar> DenseLinear<Scalar>::backward(const MatrixD<Scalar>& upstream_grad){
/*
SHAPE ASSERTIONS: 
    input:          [batch_size × input_dim]
    input_cache:    [batch_size × input_dim]
    weights:        [output_dim × input_dim]
    bias:           [output_dim × 1]
    weights^T:      [input_dim × output_dim]
    output:         [batch_size × output_dim]
    upstream_grad:  [batch_size x output_dim]
    grad_weights:   [output_dim x input_dim]
    grad_bias:      [output_dim x 1]
    downstream_grad:[batch_size x input_dim]    
*/
    VERBOSEL2("Inside DenseLinear::backward");
    //Validate input
    ASSERT(upstream_grad.rows() == this->input_cache.rows()); //both rows == batch_size
    ASSERT(upstream_grad.cols() == this->output_dim);
    
    //Validate Layers member sizes
    DASSERT(this->grad_weights.rows() == this->weights.rows());
    DASSERT(this->grad_weights.cols() == this->weights.cols());
    DASSERT(this->input_cache.cols() == this->weights.cols());
   
    //  = dL/dW 
    //  = dL/dy *dy/dW
    //      dL/dy = upstream_grad
    //      dy/dW = x | specifically if y_i = [sum over j (w_ij * x_i + b_i)]  then dy_i/dW_ij = x_j
    //  = upstream_grad^T * input
    this->grad_weights = upstream_grad.transpose() * this->input_cache;
    //Confirm Shape is unchanged
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
    //Confirm Shape is unchanged 
    DASSERT(this->grad_bias.size() == this->bias.size());
    
    //grad_input [B * in_d]
    //  = dL/dx 
    //  = dL/dy * dy/dx 
    //      dL/dy = upstream_grad
    //      dy/dx = W 
    //  = upstream_grad * Weights
    MatrixD<Scalar> downstream_grad = upstream_grad * this->weights; //[b_size X output_dim] * [output_dim X input_dim] = [b_size X input_dim]
    //Validate output size
    DASSERT(downstream_grad.cols() == this->input_cache.cols() && downstream_grad.rows() == input_cache.rows());
    DASSERT(downstream_grad.rows() > 0 && downstream_grad.rows() <= configV::Training__batch_size);
    DASSERT(downstream_grad.cols() == this->input_dim);
   
    VERBOSEL2("Finished DenseLienar::backward");
    return downstream_grad;
}

template <typename Scalar>
const MatrixD<Scalar>& DenseLinear<Scalar>::get_input_cache()const{
    return this->input_cache;
}

template <typename Scalar>
MatrixD<Scalar>& DenseLinear<Scalar>::get_input_cache(){
    return this->input_cache;
}