#pragma once
#include <omp.h>
#include <random>
#include "custom_types.h"
#include "config_values.h"
#include "param_init_utils.h"

template <typename Scalar>
SparseLinear<Scalar>::SparseLinear(
    unsigned int input_dim,
    unsigned int output_dim,
    InitFn init_fn 
):
    input_dim(input_dim),
    output_dim(output_dim)
{
    std::mt19937 gen(configV::Global__seed);
    
    this->weights = MatrixD<Scalar>(output_dim, input_dim);
    this->weights_grad = MatrixD<Scalar>(output_dim, input_dim);
    this-> bias = VectorD<Scalar>::Zero(output_dim);
    this-> bias_grad = VectorD<Scalar>::Zero(output_dim);
    
    for (size_t i = 0; i < output_dim; ++i){
        this->bias(0,i) = init_fn(input_dim, output_dim, gen); //TODO: confirm order of args is correct
        for (size_t j = 0; j < input_dim; ++j){
            this->weights(i,j) = init_fn(input_dim, output_dim, gen);
        }
    }
    
}

template <typename Scalar>
MatrixD<Scalar> SparseLinear<Scalar>::forward(const Batch<Scalar>& input){
    this-> input_cache = input;
    
    const unsigned int batch_size = static_cast<int>(input.size());
    MatrixD<Scalar> out(batch_size, this->output_dim); //Pre-allocate MatrixD to populate with SSR forward ouput

    #pragma omp parallel for
    for(size_t i =0; i < batch_size; ++i){
        VectorD<Scalar> ssr_output = this->bias;
        for (int j = 0; j < input[i].nnz; ++j){
            int idx = input[i].indices[j];
            Scalar val = input[i].data[j];
            ssr_output += val * this->weights.row(idx).transpose();
        }
        out.row(i) = ssr_output;
    }
    return out;
}


template <typename Scalar>
MatrixD<Scalar> SparseLinear<Scalar>::backward(const MatrixD<Scalar>& upstream_grad){
    const int batch_size = static_cast<int>(this->input_cache.size());
    MatrixD<Scalar> downstream_grad(batch_Size, this->input_dim);
   
    this->grad_weights.setZero();
    this->grad_bias.setZero();
    
    //gradient of loss wrt bias
    // = dL/db 
    // = dL/dy * dy/db 
    //     dL/dy = upstream_grad    
    //     dy/db = I 
    // = upstream_grad * I
    this->grad_bias = upstream_grad.colwise().sum().transpose(); //Colwise sum because bias VectorD, input MatrixD


    //Thread local grad_weights
    const int num_threads;
    std::vector<MatrixD<Scalar>> thread_local_weights_grad(num_threads, MatrixD<Scalar>::Zero(this->output_dim, this->input_dim));
    
    //grad_weights [out_d * in_d]
    //  = dL/dW 
    //  = dL/dy *dy/dW
    //      dL/dy = upstream_grad
    //      dy/dW = x | specifically if y_i = [sum over j (w_ij * x_i + b_i)]  then dy_i/dW_ij = x_j
    //  = upstream_grad^T * input
    #pragma omp parallel for
    for(int i = 0; i < batch_size; ++i){
        const SingleSparseRow<Scalar>& row = this->input_cache[i];
        const VectorD<Scalar> upstream_row = upstream_grad.row(i).transpose();
        
        MatrixD<Scalar>& local_grad = thread_local_weights_grad[omp_get_thread_num()];
        for(size_t j = 0; j < row.nnz; ++j){
            int col = row.indices[j];
            Scalar val = row.data[j];

            local_grad.col(col) += upstream_row * val;
        }

        downstream_grad.row(i) = (this->weights * upstream_row).transpose();
    }
    //reduce thread-local grad_weights
    for (const auto& local : thread_local_weights_grad){
        this->grad_weights += local;
    }
    return downstream_grad;
}

bool SparseLinear<Scalar>::supports_sparse_input() const {return  true;}
bool SparseLinear<Scalar>::has_trainable_params() const {return true;}
void SparseLinear<Scalar>::zero_grad(){
    this->grad_weights.setZero();
    this->grad_bias.serZero();
}
//Getters

//Getters - weights
template <typename Scalar>
MatrixD<Scalar>& SparseLinear<Scalar>::get_weights(){
    return  this->weights;
}

template <typename Scalar>
const MatrixD<Scalar>& SparseLinear<Scalar>::get_weights() const{
    return  this->weights;
}

template <typename Scalar>
const MatrixD<Scalar>& SparseLinear<Scalar>::get_grad_weights() const{
    return  this->grad_weights;
}

//Getters - biases

template <typename Scalar>
VectorD<Scalar>& SparseLinear<Scalar>::get_bias(){
    return  this->bias;
}

template <typename Scalar>
const VectorD<Scalar>& SparseLinear<Scalar>::get_bias() const{
    return  this->bias;
}

template <typename Scalar>
const VectorD<Scalar>& SparseLinear<Scalar>::get_grad_bias() const{
    return  this->grad_bias;
}

template <typename Scalar>
const Batch<Scalar>& SparseLinear<Scalar>::get_input_cache() const{
    return this->input_cache;
}

