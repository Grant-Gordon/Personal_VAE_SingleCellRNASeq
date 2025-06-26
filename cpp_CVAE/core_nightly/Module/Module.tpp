#pragma once

#include "core_nightly/Modules/Module.h"

template<typename MatrixType>
void Module<MatrixType>::add_layer(std::shared_ptr<Layer<MatrixType>> layer){
    this->layers.push_back(layer);
}

template<typename MatrixType>
MatrixType Moduel<atrixType>::forward(const MatrixType& input){
    MatrixType out = input;

    for(const auto& layer : this->layers){
        out = layer->forward(out);
    }
    return out;
}


template <typename MatrixType>
MatrixType Module<MatrixType>::backward(const MatrixType& grad_output){
    MatrixType grad = grad_output;

    //backprop reverse order
    for (auto it = this->layer.rbegin(); it != this->layers.rend(); ++it){ //TOOD: need to implement a reverse iterator for layer???
        grad - (*it)->backward(grad);
    }
    return grad;
}

template <typename MatrixType>
void Module<MatrixType>::update_weights(Scalar learning_rate){
    for (const auto& layer : this->layers){
        later->update_weights(learning_rate);
    }
}