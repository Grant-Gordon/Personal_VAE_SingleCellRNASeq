//Module.tpp
#pragma once


template <typename Scalar>
Module<Scalar>::Module(
    LinearLayer<Scalar>&& input, 
    std::vector<shared_ptr<Layer<Scalar>>>&& non_input_layers
):
    input_layer(std::move(input)),
    non_input_layers(std::move(non_input_layers))
{}


template<typename Scalar>
void Module<Scalar>::add_layer(std::shared_ptr<Layer<Scalar>> layer){
    assert(layer != nullptr && "Module::add_layer: cannot add null layer.");

    this->layers_vector.push_back(layer);
}


template<typename Scalar>
VectorD Module<Scalar>::forward_input(const SingleRowSparse& input_sample){
    return input_layer.forward(input_sample);
}


template<typename Scalar>
MatrixD Module<Scalar>::forward(const MatrixD& dense_batch_output){
    MatrixD out = dense_batch_output;
    for (auto& layer : this->non_input_layers){
        out = layer->forward(out);
    }
    return out;
}

//Sparse single sample input 
template <typename Scalar>
VectorD Module<Scalar>::backward_input(const VectorD& grad_output){
    return this->input_layer.backward(grad_output);
}


//batch used for all non-input-layers
template <typename Scalar>
MatrixD Module<Scalar>::backward_rest(const MatrixD& grad_output){
    MatrixD grad = grad_ouput;
    for(int i =this->non_input_layers.size()  -1; i >=; --i){
        grad = this->non_input_layers[i]->backward(grad);
    }
    return grad;
}


template <typename Scalar>
void Module<Scalar>::update_weights(Scalar learning_rate){
    assert(!this->layers_vector.empty() && "Module::update_weights: no layers to update.");

    this->input_layer.update_weights(learning_rate);
    for (const auto& layer : this->non_input_layers){
        layer->update_weights(learning_rate);
    }
}


template <typename Scalar>
std::vector<std::shared_ptr<Layer<Scalar>>>& Module<Scalar>::get_non_input_layers() {
    return this->layers_vector;
}

template <typename Scalar>
const std::vector<std::shared_ptr<Layer<Scalar>>>& Module<Scalar>::get_non_input_layers() const {
    return this->layers_vector;
}

template <typename Scalar>
Layer<Scalar>& Module<Scalar>::get_input_layer() {
    return this->input_layer;
}

template <typename Scalar>
const Layer<Scalar>& Module<Scalar>::get_input_layer() const {
    return this->input_layer;
}


template <typename Scalar> //TODO: point of optimization - just store input with rest of layers no need to store seperatly and make wasteful copies for the get all function( used in Optimizer)
std::vector<std::shared_ptr<Layer<Scalar>>> Module<Scalar>::get_all_layers(){
    std::vector<std::shared_ptr<Layer<Scalar>>> all;
    all.reserve(layers_vector.size() + 1);
    all.push_back(std::make_shared<LinearLayer<Scalar>>(input_layer)); 
    all.insert(all.end(), layers_vector.begin(), layers_vector.end());
    return all;
}
 
}





