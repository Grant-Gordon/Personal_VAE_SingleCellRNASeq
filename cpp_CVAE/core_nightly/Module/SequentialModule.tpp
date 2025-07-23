//SequentialModule.tpp
#pragma once
#include <vector>
#include <stdexcept>
#include "custom_types.h"
#include "Layer.h"
template<typename Scalar>
SequentialModule<Scalar>::SequentialModule(
    std::vector<std::shared_ptr<Layer<Scalar>>>&& layers_vector
){
    this->layers_vector = std::move(layers_vector);
}

//Dense input
template<typename Scalar>
MatrixD<Scalar> SequentialModule<Scalar>::forward(const MatrixD<Scalar>& input){
    MatrixD<Scalar> out = input;
    for (auto& layer : this->layers_vector){
        out = layer->forward(out);
    }
    return out;
}

//Sparse input batch
template<typename Scalar>
MatrixD<Scalar> SequentialModule<Scalar>::forward(const Batch<Scalar>& input){
    if(!this->supports_sparse_input()){
        throw runtime_error("Sparse input not supported by this SequentialModule");
    }
    MatrixD<Scalar> out = this->layers_vector[0]->forward(input);
    for(size_t i = 1; i < this->layers_vector.size(); ++i){
        out=this->layers_vector[i]->forward(out);
    }
    return out;
}


template<typename Scalar>
MatrixD<Scalar> SequentialModule<Scalar>::backward(const MatrixD<Scalar>& upstream_grad){
    MatrixD<Scalar> downstream_grad = upstream_grad;
    for(int i = static_cast<int>(this->layers_vector.size()) -1; i >=0; --i){
        downstream_grad = this->layers_vector[i]->backward(downstream_grad);
    }
    return downstream_grad;
}

template<typename Scalar>
bool SequentialModule<Scalar>::supports_sparse_input() const{
    return !this->layers_vector.empty() && this->layers_vector[0]->supports_sparse_input();
}

template<typename Scalar>
void SequentialModule<Scalar>::zero_grad(){
    for (auto& layer : this->layers_vector){
        if(layer->has_trainable_params()){
            layer->zero_grad();
        }
    }
}
