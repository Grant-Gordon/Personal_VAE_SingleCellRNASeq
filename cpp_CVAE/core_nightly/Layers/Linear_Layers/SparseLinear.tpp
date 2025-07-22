#pragma once
#include <omp.h>
#include <random>
#include <vector>
#include "custom_types.h"
#include "config_values.h"
#include "param_init_utils.h"
#include "macros.h"

template <typename Scalar>
SparseLinear<Scalar>::SparseLinear(
    unsigned int input_dim,
    unsigned int output_dim,
    InitFn<Scalar> init_fn 
):
    input_dim(input_dim),
    output_dim(output_dim)
{
    std::mt19937 gen(configV::Global__seed);
    
    this->weights = MatrixD<Scalar>(output_dim, input_dim);
    this->weights_grad = MatrixD<Scalar>(output_dim, input_dim);
    this-> bias = VectorD<Scalar>::Zero(output_dim);
    this-> bias_grad = VectorD<Scalar>::Zero(output_dim);
    
    ASSERT(this->input_dim > 0 && this->output_dim > 0);
    ASSERT(this->weights.rows() == this->output_dim && this->weights.cols() == this->inpu_dim);
    ASSERT(this->bias.size() == output_dim);



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
MatrixD<Scalar> SparseLinear<Scalar>::forward(const Batch<Scalar>& input){
    ASSERT(!input.empty());
    ASSERT(this->weights.rows() == this->bias.size());
    this-> input_cache = input;
    
    const unsigned int batch_size = static_cast<int>(input.size());
    MatrixD<Scalar> out(batch_size, this->output_dim); //Pre-allocate MatrixD to populate with SSR forward ouput

    #pragma omp parallel for
    for(size_t i =0; i < batch_size; ++i){
        VectorD<Scalar> ssr_output = this->bias;
        ASSERT(input[i].indices.size() == input[i].data.size());
        for (int j = 0; j < input[i].nnz; ++j){
            ASSERT(std::isfinite(input[i].data[j]));
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
    ASSERT(upstream_grad.rows() == static_cast<int>(this->input_cache.size()));
    ASSERT(upstream_grad.cols() == this->output_dim);

    const int batch_size = static_cast<int>(this->input_cache.size());
    MatrixD<Scalar> downstream_grad(batch_size, this->input_dim);
   
    this->get_grad_weights().setZero();
    this->get_grad_bias().setZero();
    ASSERT(this->get_grad_weights().rows() == this->get_weights().rows());
    ASSERT(this->get_grad_weights().cols() == this->get_weights().cols());
    ASSERT(this->get_grad_bias().size() == this->get_bias().size());

    
    //gradient of loss wrt bias
    // = dL/db 
    // = dL/dy * dy/db 
    //     dL/dy = upstream_grad    
    //     dy/db = I 
    // = upstream_grad * I
    this->grad_bias = upstream_grad.colwise().sum().transpose(); //Colwise sum because bias VectorD, input MatrixD


    //Thread local grad_weights
    const int num_threads = omp_get_max_threads(); //TODO assign macro for available threads (i.e. want to reserve some threads for batch loading etc)
    ASSERT(num_threads > 0);
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
        ASSERT(row.nnz <= row.indices.size() && row.nnz <= row.data.size());
        for(size_t j = 0; j < row.nnz; ++j){
            int col = row.indices[j];
            ASSERT(col >= 0 && col < this->input_dim);
            
            Scalar val = row.data[j];
            ASSERT(std::isfinite(val));

            local_grad.col(col) += upstream_row * val;
            ASSERT(local_grad.cols() > col);
        }

        downstream_grad.row(i) = (this->get_weights() * upstream_row).transpose();
        ASSERT((this->get_weights() * upstream_row).rows() == this->output_dim);

    }
    //reduce thread-local grad_weights
    for (const auto& local : thread_local_weights_grad){
        this->get_grad_weights()+= local;
    }
    return downstream_grad;
}


template <typename Scalar>
void SparseLinear<Scalar>::zero_grad(){
    this->get_weights().setZero(); //TODO: confirm this will change the reference not just make some random copy of zero matrix
    this->get_grad_bias().setZero();
}
