//Adam.tpp
//AI Acknowledgement - This File utilized code from ChatGPT


#pragma once
#include <cmath>
#include <cassert>
#include "LinearLayer.h"

template <typename Scalar>
Adam<Scalar>::Adam(
    Scalar learning_rate,
    Scalar beta1,
    Scalar beta2,
    Scalar epsilon
) : learning_rate(learning_rate),
    beta1(beta1),
    beta2(beta2),
    epsilon(epsilon),
    timestep(0)
{
    static_assert(std::is_floating_point<Scalar>::value, "Adam: Scalar must be floating-point.");
    assert(learning_rate > 0 && "Adam: learning rate must be > 0.");
    assert(beta1 >= 0 && beta1 < 1 && "Adam: beta1 must be in [0, 1).");
    assert(beta2 >= 0 && beta2 < 1 && "Adam: beta2 must be in [0, 1).");
    assert(epsilon > 0 && "Adam: epsilon must be > 0.");
}

template <typename Scalar>
void Adam<Scalar>::step(std::vector<std::shared_ptr<Layer<Scalar>>>& layers_vector) {
    assert(!layers_vector.empty() && "Adam::step: layers_vector is empty.");

    ++this->timestep;

    for (auto& layer : layers_vector) {
        auto* raw_layer = layer.get();
        auto* linear = dynamic_cast<LinearLayer<Scalar>*>(raw_layer); //TODO: confirm this isn't turning all layers into Linear (what about RELU)
        if (!linear) continue;  // skip non-trainable layers

        // === WEIGHTS ===
        MatrixD& weights = linear->get_weights();
        const MatrixD& grad_weights = linear->get_grad_weights();

        ParamState& w_state = weight_state[raw_layer];
        if (w_state.m.size() == 0) {
            w_state.m = MatrixD::Zero(grad_weights.rows(), grad_weights.cols());
            w_state.v = MatrixD::Zero(grad_weights.rows(), grad_weights.cols());
        }

        // Update moments
        w_state.m = beta1 * w_state.m + (1 - beta1) * grad_weights;
        w_state.v = beta2 * w_state.v + (1 - beta2) * grad_weights.cwiseProduct(grad_weights);

        // Bias correction
        MatrixD m_hat = w_state.m / (1 - std::pow(beta1, timestep));
        MatrixD v_hat = w_state.v / (1 - std::pow(beta2, timestep));

        // Weight update
        weights -= (learning_rate * m_hat.array() / (v_hat.array().sqrt() + epsilon)).matrix();

        // === BIASES ===
        VectorD& bias = linear->get_bias();
        const VectorD& grad_bias = linear->get_grad_bias();

        ParamState& b_state = bias_state[raw_layer];
        if (b_state.m.size() == 0) {
            b_state.m = VectorD::Zero(grad_bias.size());
            b_state.v = VectorD::Zero(grad_bias.size());
        }

        b_state.m = beta1 * b_state.m + (1 - beta1) * grad_bias;
        b_state.v = beta2 * b_state.v + (1 - beta2) * grad_bias.cwiseProduct(grad_bias);

        VectorD m_hat_b = b_state.m / (1 - std::pow(beta1, timestep));
        VectorD v_hat_b = b_state.v / (1 - std::pow(beta2, timestep));

        bias -= (learning_rate * m_hat_b.array() / (v_hat_b.array().sqrt() + epsilon)).matrix();
    }
}
