//Adam.tpp
//AI Acknowledgement - This File utilized code from ChatGPT


#pragma once
#include <cmath>
#include <cassert>
#include <omp.h>
#include "config_values.h"
#include "custom_types.h"
#include "Layer_all.h"

template <typename Scalar>
Adam<Scalar>::Adam( int beta1, int beta2, int epsilon
): 
    this->timestep(0),
    this->beta1(beta1),
    this->(beta2),
    this->epsilon(epsilon)
{}

template <typename Scalar>
void Adam<Scalar>::step(std::vector<std::shared_ptr<Layer<Scalar>>>& layers_vector) {
    ++this->timestep;
    #pragma omp parallel for
    for (auto& layer : layers_vector) {
        if (!layer->has_trainable_params()) continue; //Only train on layers with trainable params (e.g. skips RELU)

        auto trainable = std::dynamic_pointer_cast<TrainableLayer<Scalar>>(layer);
        // === WEIGHTS ===
        MatrixD<Scalar>& weights = trainable->get_weights();
        const MatrixD<Scalar>& grad_weights = trainable->get_grad_weights();

        ParamState& w_state = weight_state[trainable.get()];
        if (w_state.m.size() == 0) {
            w_state.m = MatrixD<Scalar>::Zero(grad_weights.rows(), grad_weights.cols());
            w_state.v = MatrixD<Scalar>::Zero(grad_weights.rows(), grad_weights.cols());
        }

        // Update moments
        w_state.m = this->beta1 * w_state.m + (1 - this->beta1) * grad_weights;
        w_state.v = this->beta2 * w_state.v + (1 - this->beta2) * grad_weights.cwiseProduct(grad_weights);

        // Bias correction
        MatrixD<Scalar> m_hat = w_state.m / (1 - std::pow(this->beta1, timestep));
        MatrixD<Scalar> v_hat = w_state.v / (1 - std::pow(this->beta2, timestep));

        // Weight update
        weights -= (configV::Training__lr * m_hat.array() / (v_hat.array().sqrt() + this->epsilon)).matrix();

        // === BIASES ===
        VectorD<Scalar>& bias = trainable->get_bias();
        const VectorD<Scalar>& grad_bias = trainable->get_grad_bias();

        ParamState& b_state = bias_state[trainable.get()];
        if (b_state.m.size() == 0) {
            b_state.m = VectorD<Scalar>::Zero(grad_bias.size());
            b_state.v = VectorD<Scalar>::Zero(grad_bias.size());
        }

        b_state.m = this->beta1 * b_state.m + (1 - this->beta1) * grad_bias;
        b_state.v = this->beta2 * b_state.v + (1 - this->beta2) * grad_bias.cwiseProduct(grad_bias);

        VectorD<Scalar> m_hat_b = b_state.m / (1 - std::pow(this->beta1, timestep));
        VectorD<Scalar> v_hat_b = b_state.v / (1 - std::pow(this->beta2, timestep));

        bias -= (configV::Training__lr * m_hat_b.array() / (v_hat_b.array().sqrt() + this->epsilon)).matrix();
    }
}