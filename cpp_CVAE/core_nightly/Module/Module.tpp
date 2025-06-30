//Module.tpp
#pragma once

template<typename MatrixType>
void Module<MatrixType>::add_layer(std::shared_ptr<Layer<MatrixType>> layer){
    assert(layer != nullptr && "Module::add_layer: cannot add null layer.");

    this->layers_vector.push_back(layer);
}

template<typename MatrixType>
MatrixType Module<MatrixType>::forward(const MatrixType& input){
    assert(!this->layers_vector.empty() && "Module::forward: no layers in module.");

    MatrixType out = input;

    for(const auto& layer : this->layers_vector){
        out = layer->forward(out);
    }
    return out;
}


template <typename MatrixType>
MatrixType Module<MatrixType>::backward(const MatrixType& grad_output){
    assert(!this->layers_vector.empty() && "Module::backward: no layers to backprop through.");

    MatrixType grad = grad_output;

    //backprop reverse order
    for (auto it = this->layers_vector.rbegin(); it != this->layers_vector.rend(); ++it){ //TOOD: need to implement a reverse iterator for layer???
        grad = (*it)->backward(grad);
    }
    return grad;
}

template <typename MatrixType>
void Module<MatrixType>::update_weights(typename MatrixType::Scalar learning_rate){
    assert(!this->layers_vector.empty() && "Module::update_weights: no layers to update.");

    for (const auto& layer : this->layers_vector){
        layer->update_weights(learning_rate);
    }
}

template <typename MatrixType>
std::vector<std::shared_ptr<Layer<MatrixType>>>& Module<MatrixType>::get_layers(){
    return this->layers_vector;
} 

template <typename MatrixType>
const std::vector<std::shared_ptr<Layer<MatrixType>>>& Module<MatrixType>::get_layers() const{
    return this->layers_vector;
}
