//Adam.tpp
//AI Acknowledgement - This File utilized code from ChatGPT
#pragma once
#include <cmath>
#include "LinearLayer.h"


template <typename MatrixType>
Adam<MatrixType>::Adam(
    typename MatrixType::Scalar learning_rate,
    typename MatrixType::Scalar beta1, 
    typename MatrixType::Scalar beta2,
    typename MatrixType::Scalar epsilon
):
    learning_rate(learning_rate),
    beta1(beta1),
    beta2(beta2),
    epsilon(epsilon),
    timestep(0) 
    {}


template <typename MatrixType>
void Adam<MatrixType>::step(std::vector<std::shared_ptr<Layer<MatrixType>>>& layers_vector) {
    ++this->timestep;

    for (auto& layer: layers_vector){
        //Skip Layers that dont expose parameter APIS?
        auto* raw_layer = layer.get();
        auto* linear = dynamic_cast<LinearLayer<MatrixType>*>(raw_layer);
        if (!linear) continue;


        //WEIGHTS
        MatrixType& weights = linear->get_weights();
        const MatrixType& d_weights= linear->get_grad_weights();
        
        ParamState& w_state = this->weight_state[raw_layer];
        if(w_state.m.size() == 0){
            w_state.v = MatrixType::Zero(d_weights.rows(), d_weights.cols());
            w_state.m = MatrixType::Zero(d_weights.rows(), d_weights.cols());
        }

        //Update Moments
        w_state.m = beta1 * w_state.m + (1 - beta1) * d_weights;
        w_state.v = beta1 * w_state.v + (1 - beta2) * d_weights.cwiseProduct(d_weights);
        
        //Bias Correction
        MatrixType m_hat = w_state.m / (1 - std::pow(beta1, timestep));
        MatrixType v_hat = w_state.v / (1 - std::pow(beta2, timestep));

        //Weight update
        weights -= learning_rate * m_hat.array() / (v_hat.array().sqrt() + epsilon); //TODO: consider protecting v_hat.array().sqrt()?

        //BIAS 
        MatrixType& biases = linear->get_bias();
        const MatrixType& d_biases = linear->get_grad_bias();
        
        ParamState& b_state = bias_state[raw_layer];
        if (b_state.m.size() ==0){
            b_state.m = MatrixType::Zero(d_biases.rows(), d_biases.cols());
            b_state.v = MatrixType::Zero(d_biases.rows(), d_biases.cols());
        }
        b_state.m = beta1 * b_state.m + (1 - beta1) * d_biases;
        b_state.v = beta2 * b_state.v + (1 - beta2) * d_biases.cwiseProduct(d_biases);

        
        MatrixType m_hat_b = b_state.m / (1 -std::pow(beta1, timestep));
        MatrixType v_hat_b = b_state.v / (1 - std::pow(beta2, timestep));

        biases-= learning_rate * m_hat_b.array() / (v_hat_b.array().sqrt() + epsilon);
    }

}