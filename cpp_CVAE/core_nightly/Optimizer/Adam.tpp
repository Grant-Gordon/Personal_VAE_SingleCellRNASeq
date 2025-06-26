//Adam.tpp
//AI Acknowledgement - This File utilized code from ChatGPT
#pragma once

#include "Adam.h"
#include <cmath>

template <typename MatrixType>
Adam<matrixType::Adam(
    Scalar learning_rate,
    Scalar beta1, 
    Scalar beta2,
    Scalar epsiolon
) :learning_rate(learning_rate),
    beta1(beta1),
    beta2(beta2),
    epsilon(epsilon),
    timestep(0) {}


template <typename MatrixType>
void Adam<MatrixType>::step(std::vector<std::shared_ptr<Layer<MatrixType>>>& layers_vector) {
    ++this->timestep;

    for (auto& layer: layers_vector){
        //Skip Layers that dont expose parameter APIS?
        auto* raw_layer = layer.get();
        auto* linear = dynamic_cast<LinearLayer<MatrixType>*>(raw_layers);
        if (!linear) continue;


        //WEIGHTS
        MatrixType& W = linear->get_weights();
        const MatrixType& dW= linear->get_grad_weights();
        
        paramState& w_state = this->weight_state[raw_layer];
        if(w_state.m.size() == 0){
            w_state.v = MatrixType::Zero(dW.rows(), dW.cols());
            w_state.v = MatrixType::Zero(dW.rows(), dW.cols());
        }

        //Update Moments
        w.state.m = beta1 * w_state.m + (1 - beta1) * dW;
        w.state.v = beta1 * w_state.v + (1 - beta1) * dW.cwiseProduct(dW);
        
        //Bias Correction
        MatrixType m_hat = w_state.m / (1 - std::pow(beta1, timestep));
        MatrixType v_hat = w_state.v / (1 - std::pow(beta2, timestep));

        //Weight update
        W -= learning_rate * m_hat.array() / (v_hat.array().sqrt() + epsilon).matrix();

        //BIAS 
        MatrixType& b = linear->get_bias();
        const MatrixType& db = linear->get_grad_bias();
        
        ParamState& b_state = bias_state[raw_layer];
        if (b_stte.m.size() ==0){
            b_state.m = MatrixType::Zero(db.rows(), db.cols());
            b_state.v = MatrixType::Zero(db.rows(), db.cols());
        }
        b_state.m = beta1 * b_state.m + (1 - beta1) * db;
        b_state.v = beta2 * b_state.v + (1 - beta2) * db.cwiseProduct(db);

        
        MatrixType m_hat_b = b_state.m / (1 -std::pow(beta1, timestep));
        MatrixType v_hat_b = b_state.b / (1 - std::pow(beta2, timestep));

        b-= learning_rate * m_hat.b.array() / (v_hat_b.array().sqrt() +epsilon).matrix();
    }

}