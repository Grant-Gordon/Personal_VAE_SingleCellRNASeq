//Module.tpp
#pragma once

#include "config_values.h" //TODO: only used once and not positive it should be used at all. 
#include "custom_types.h"


template <typename Scalar>
Module<Scalar>::Module(
    std::vector<std::shared_ptr<Layer<Scalar>>>&& layers_vector
):
    layers_vector(std::move(layers_vector))
{}


template<typename Scalar>
void Module<Scalar>::add_layer(std::shared_ptr<Layer<Scalar>> layer){
    assert(layer && "Module::add_layer: cannot add null layer.");

    this->layers_vector.push_back(layer);
}


//Unified forward pass that handles in parallel the SSR inputs, then sequentially forwards the batch created by the SSR input to the remainder layers
template <typename Scalar>
MatrixD<Scalar> Module<Scalar>::forward(const Batch<Scalar>& batch){
    MatrixD<Scalar> out = layer[0]->forward(batch);
    for(int i = 1; i < static_cast<int>(this->layers_vector.size()); ++i){
       MatrixD<Scalar> out = this->layers_vector[i]->forward(out); //TODO does this recreate out or is this smart moving or something?
    }
    return out;
}


// Unified backprop first passing through all tayers, then parallizes the batch for the SSR input layer NOTE: because of the critical section in LinearLayer::forward(SSR) this is not actually parallelized. 
template <typename Scalar>
MatrixD<Scalar> Module<Scalar>::backward(const MatrixD<Scalar> upstream_grad, const Batch<Scalar>& batch_input){ //TODO: why am I passing inputs in? cant this be gotten elsewhere? need to define where ownership of SSR batch lives
    MatrixD<Scalar> grad = upstream_grad;
    //backprop through dense layers in reverse
    for (int i = static_cast<int>(this->layers_vector.size()); i >0; --i){
        grad = this->layers_vector[i]->backward(grad);
    }

}

template <typename Scalar>
void Module<Scalar>::update_weights(){
    assert(!this->layers_vector.empty() && "Module::update_weights: no layers to update.");

    for (const auto& layer : this->layers_vector){
        layer->update_weights(configV::Training__lr); //TODO: confirm that ADAM shouldn't be touching this. 
    }
}


template <typename Scalar>
std::vector<std::shared_ptr<Layer<Scalar>>>& Module<Scalar>::get_layers() {
    return this->layers_vector;
}

template <typename Scalar>
const std::vector<std::shared_ptr<Layer<Scalar>>>& Module<Scalar>::get_layers() const {
    return this->layers_vector;
}


template <typename Scalar>
const int Module<Scalar>::get_input_dim() const {
    auto linear_ptr = dynamic_cast<LinearLayer<Scalar>*>(this->layers_vector[0].get());
    if (!linear_ptr) {
        throw std::runtime_error("First layer must be LinearLayer to get input_dim.");
    }
    return linear_ptr->input_dim;
}

template <typename Scalar>
const int Module<Scalar>::get_output_dim() const {
    auto linear_ptr = dynamic_cast<LinearLayer<Scalar>*>(this->layers_vector.back().get());
    if (!linear_ptr) {
        throw std::runtime_error("Last layer must be LinearLayer to get output_dim.");
    }
    return linear_ptr->output_dim;
}
