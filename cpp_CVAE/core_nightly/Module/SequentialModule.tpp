//SequentialModule.tpp
#pragma once
#include <vector>
#include <stdexcept>
#include "custom_types.h"
#include "Layer.h"
#include "macros.h"
template<typename Scalar>
SequentialModule<Scalar>::SequentialModule(
    const std::vector<std::shared_ptr<Layer<Scalar>>> layers_vector
):
layers_vector(layers_vector)
{
    ASSERT(!layers_vector.empty());
    for(size_t i =0; i < layers_vector.size(); ++i){
        ASSERT(layers_vector[i] != nullptr);
    }
}

//Dense input
template<typename Scalar>
MatrixD<Scalar> SequentialModule<Scalar>::forward(const MatrixD<Scalar>& input){
    ASSERT(input.rows() > 0 && input.cols() > 0);
    MatrixD<Scalar> out = input;
   
    for (size_t i = 0; i < this->layers_vector.size(); ++i){
        out = this->layers_vector[i]->forward(out);
        //Validate output Shape
        DASSERT(out.rows() > 0 && out.cols() > 0);
        DASSERT(out.allFinite());
    }
    return out;
}

//Sparse input batch
template<typename Scalar>
MatrixD<Scalar> SequentialModule<Scalar>::forward(const Batch<Scalar>& input){
    ASSERT(!this->layers_vector.empty());
    if(!this->supports_sparse_input()){
        throw std::runtime_error("Sparse input not supported by this SequentialModule");
    }

    MatrixD<Scalar> out = this->layers_vector[0]->forward(input);
    DASSERT(out.rows() > 0 && out.cols() > 0);

    for(size_t i = 1; i < this->layers_vector.size(); ++i){
        out=this->layers_vector[i]->forward(out);
        DASSERT(out.rows() > 0 && out.cols() >0);
        DASSERT(out.allFinite());
    }
    return out;
}


template<typename Scalar>
MatrixD<Scalar> SequentialModule<Scalar>::backward(const MatrixD<Scalar>& upstream_grad){
    ASSERT(upstream_grad.rows() > 0 && upstream_grad.cols() > 0);
    MatrixD<Scalar> downstream_grad = upstream_grad;

    for(int i = static_cast<int>(this->layers_vector.size()) -1; i >=0; --i){
        downstream_grad = this->layers_vector[i]->backward(downstream_grad);
        DASSERT(downstream_grad.rows() > 0 && downstream_grad.cols() > 0);
       // DASSERT(downstream_grad.allFinite());
    }
    return downstream_grad;
}

template<typename Scalar>
bool SequentialModule<Scalar>::supports_sparse_input() const{
    return !this->layers_vector.empty() && this->layers_vector[0]->supports_sparse_input();
}

