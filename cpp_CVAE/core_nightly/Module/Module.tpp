//Module.tpp
#pragma once


template <typename Scalar>
Module<Scalar>::Module(
    std::vector<shared_ptr<Layer<Scalar>>>&& layers_vector
):
    layers_vector(std::move(layers_vector))
{}


template<typename Scalar>
void Module<Scalar>::add_layer(std::shared_ptr<Layer<Scalar>> layer){
    assert(this->layers_vector != nullptr && "Module::add_layer: cannot add null layer.");

    this->layers_vector.push_back(layer);
}


//Unified forward pass that handles in parallel the SSR inputs, then sequentially forwards the batch created by the SSR input to the remainder layers
template <typename Scalar>
MatrixD Module<Scalar>::forward(const std::vector<SingleRowSparse<Scalar>>& batch){
    
    MatrixD out(batch_size, this->layers_vector[0]->output_dim);
    
    //parallelize the input layer so that each sample(SSR) in the batch gets its own thread
    #pragma omp parallel for
    for (int i = 0; i < batch.size(); ++i){
        out.row(i) = this->layers_vector[0]->forward(batch[i]); //forward takes in SSR, returns VectorD i.e. dense row 
    }

    //sequentially forward the rest of the layers
    for(int i = 1; i < this->layers_vector.size(); ++i){
        out = this->layers_vector[i]->forward(out);
    }
    
    return out;
}



// Unified backprop first passing through all tayers, then parallizes the batch for the SSR input layer NOTE: because of the critical section in LinearLayer::forward(SSR) this is not actually parallelized. 
template <typename Scalar>
MatrixD Modulce<Scalar>::backward(const MatrixD upstream_grad, const std::vector<SingleSparseRow<Scalar>>& batch_input){ //TODO: why am I passing inputs in? cant this be gotten elsewhere? need to define where ownership of SSR batch lives
    MatrixD grad = upstream_grad;
    //backprop through dense layers in reverse
    for (int i = static_cast<int>(this->layers_vector.size()) -1; i >0; --i){
        grad = this->layers_vector[i]->backward(grad);
    }
    //backprop sparse input layer per sample
    #pragma omp parallel for //NOTE: this is technically parallelized but, because backward(SingleSparseRow) contains critical section, in practice it is sequential 
    for(int i =0; i < batch_input.size(); ++i){
        grad.row(i) = this->layers_vector[0]->backward(grad.row(i).transpose(), batch_input[i]).transpose();
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
std::vector<std::shared_ptr<Layer<Scalar>>>& Module<Scalar>::get_layers() {
    return this->layers_vector;
}

template <typename Scalar>
const std::vector<std::shared_ptr<Layer<Scalar>>>& Module<Scalar>::get_layers() const {
    return this->layers_vector;
}