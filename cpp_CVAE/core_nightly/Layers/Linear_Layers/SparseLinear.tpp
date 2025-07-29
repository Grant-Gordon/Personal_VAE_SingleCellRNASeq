//SparseLinear.tpp
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
)
{
    VERBOSEL2("Inside SparseLinear::SparseLinear")
    /*
    SHAPE ASSERTIONS: 
    input:         [std::vector size == batch_Size]
    weights:       [output_dim × input_dim]
    bias:          [output_dim × 1]
    weights^T:     [input_dim × output_dim]
    grad_weights:  [output_dim × input_dim]
    grad_bias:     [output_dim × 1]
    */
   
   //Validate inputs
   ASSERT(input_dim > 0 && output_dim > 0);
   
   this->input_dim = input_dim;
   this->output_dim = output_dim;
   std::mt19937 gen(configV::Global__seed);
   
   this->weights = MatrixD<Scalar>(output_dim, input_dim);
   this->grad_weights = MatrixD<Scalar>(output_dim, input_dim);
   this->bias = VectorD<Scalar>::Zero(output_dim);
   this->grad_bias = VectorD<Scalar>::Zero(output_dim);
   
    //Validate W/b shape and dims properly assigned
    DASSERT(this->input_dim > 0 && this->output_dim > 0);
    DASSERT(this->weights.rows() == this->output_dim && this->weights.cols() == this->input_dim);
    DASSERT(this->bias.rows() == output_dim && this->bias.cols() == 1);
    
    
    for (size_t i = 0; i < output_dim; ++i){
        this->bias(i) = init_fn(input_dim, output_dim, gen);
        DASSERT(std::isfinite(this->bias(i)));
        
        for (size_t j = 0; j < input_dim; ++j){
            this->weights(i,j) = init_fn(input_dim, output_dim, gen);
            DASSERT(std::isfinite(this->weights(i,j)));
        }
    }
    
    VERBOSEL2("Finished SparseLinear::SparseLinear")
}

template <typename Scalar>
MatrixD<Scalar> SparseLinear<Scalar>::forward(const Batch<Scalar>& input){
    VERBOSEL2("Inside SparseLinear::forward");
    
    ASSERT(!input.empty());
    DASSERT(this->weights.rows() == this->output_dim);
    DASSERT(this->weights.cols() == this->input_dim);
    DASSERT(this->bias.size() == this->output_dim);

    this->input_cache_ptr = &input; //TODO cannot move a const & 
    
    const unsigned int batch_size = static_cast<unsigned int>(input.size());
    MatrixD<Scalar> out(batch_size, this->output_dim); //Pre-allocate MatrixD to populate with SSR forward ouput
    //Validate ouput dim allocation
    DASSERT(out.rows() == batch_size);
    DASSERT(out.cols() == this->output_dim);

    #pragma omp parallel for
    for(size_t i =0; i < batch_size; ++i){
        ASSERT(input[i]); //Not Null
        const SingleSparseRow<Scalar>& row = *index[i];
        ASSERT(row.nnz>= 0); //empty rows shouldny be in sparse formats

        VectorD<Scalar> ssr_output = this->bias;

        for (int j = 0; j < row.nnz; ++j){
            const Scalar val = row.data[j];
            const int idx = row.indices[j];

            DASSERT(std::isfinite(val));
            ASSERT(idx >= 0 && idx < this->input_dim);
            
            ssr_output += val * this->weights.col(idx);
        }
        out.row(i) = ssr_output.transpose(); //bias + weights  = [out;ut_dim x 1] -> row
        
    }
    VERBOSEL2("Finisehd SparseLinear::forward");
    return out;
}


template <typename Scalar>
MatrixD<Scalar> SparseLinear<Scalar>::backward(const MatrixD<Scalar>& upstream_grad){
    VERBOSEL2("Inside SparseLinear::backward");
   
    const int batch_size = static_cast<int>(this->input_cache_ptr->size());//TODO maybe need to derefernce input-cache_ptr?
   //Validate input shape
    ASSERT(upstream_grad.rows() == batch_size);
    ASSERT(upstream_grad.cols() == this->output_dim);

    MatrixD<Scalar> downstream_grad(batch_size, this->input_dim);
    this->grad_weights.setZero();//TODO confirm these are initialized somwhere? or if they need to be?
    this->grad_bias.setZero();
   
    DASSERT(this->grad_weights.rows() == this->weights.rows());
    DASSERT(this->grad_weights.cols() == this->weights.cols());
    DASSERT(this->grad_bias.size() == this->bias.size());

    
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
        const SingleSparseRow<Scalar>& row = *(*this->input_cache_ptr)[i]; //batch stores unique_ptrs so need to dereference 
        const VectorD<Scalar> upstream_row = upstream_grad.row(i).transpose();
        MatrixD<Scalar>& local_grad = thread_local_weights_grad[omp_get_thread_num()];
        
        for(size_t j = 0; j < row.nnz; ++j){
            const int col = row.indices[j];
            ASSERT(col >= 0 && col < this->input_dim);
            
            Scalar val = row.data[j];
            ASSERT(std::isfinite(val));

            local_grad.col(col) += upstream_row * val;
        }

        downstream_grad.row(i) = (upstream_grad.transpose() * this->weights); //upstream_grad^T * W = [1 x output_dim] * [outputdim x 1]
        DASSERT(dowstream_grad.col(i) == this->input_dim);
        DASSERT(dowstream_grad.row(i) == 1);

    }
    //reduce thread-local grad_weights
    for (const auto& local : thread_local_weights_grad){
        this->grad_weights+= local;
    }
    VERBOSEL2("Finished SparseLinear::backward");
    return downstream_grad;
}

template <typename Scalar>
Batch<Scalar>& SparseLinear<Scalar>::get_input_cache(){
    return *this->input_cache_ptr;
}

template <typename Scalar>
const Batch<Scalar>& SparseLinear<Scalar>::get_input_cache()const{
    return *this->input_cache_ptr;
}

template <typename Scalar>
const Batch<Scalar>* SparseLinear<Scalar>::get_input_cache_ptr() const{
    return this->input_cache;
}