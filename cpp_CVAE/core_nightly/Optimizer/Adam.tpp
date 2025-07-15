//Adam.tpp
//AI Acknowledgement - This File utilized code from ChatGPT


#pragma once
#include <cmath>
#include <cassert>
#include "LinearLayer.h"
#include "config.h"
#include "custom_types.h"

template <typename Scalar>
Adam<Scalar>::Adam(
): 
    timestep(0)
{
    static_assert(std::is_floating_point<Scalar>::value, "Adam: Scalar must be floating-point.");
    assert(config::training_learning_rate > 0 && "Adam: learning rate must be > 0.");
    assert(config::Optim_beta1 >= 0 && config::Optim_beta1 < 1 && "Adam: config::Optim_beta1 must be in [0, 1).");
    assert(config::Optim_beta2 >= 0 && config::Optim_beta2 < 1 && "Adam: config::Optim_beta2 must be in [0, 1).");
    assert(config::Optim_epsilon > 0 && "Adam: config::Optim_epsilon must be > 0.");
}

template <typename Scalar>
void Adam<Scalar>::step(std::vector<std::shared_ptr<Layer<Scalar>>>& layers_vector) {
    assert(!layers_vector.empty() && "Adam::step: layers_vector is empty.");

    ++this->timestep;

    for (auto& layer : layers_vector) {
        if (!layer->has_trainable_params()) continue; //Only train on layers with trainable params (e.g. skips RELU)
        auto* linear = dynamic_cast<LinearLayer<Scalar>*>(layer.get());
        assert(linear && "Adam::step: expected trainable layer to be LinearLayer");
        // === WEIGHTS ===
        MatrixD<Scalar>& weights = linear->get_weights();
        const MatrixD<Scalar>& grad_weights = linear->get_grad_weights();

        ParamState& w_state = weight_state[raw_layer];
        if (w_state.m.size() == 0) {
            w_state.m = MatrixD<Scalar>::Zero(grad_weights.rows(), grad_weights.cols());
            w_state.v = MatrixD<Scalar>::Zero(grad_weights.rows(), grad_weights.cols());
        }

        // Update moments
        w_state.m = config::Optim_beta1 * w_state.m + (1 - config::Optim_beta1) * grad_weights;
        w_state.v = config::Optim_beta2 * w_state.v + (1 - config::Optim_beta2) * grad_weights.cwiseProduct(grad_weights);

        // Bias correction
        MatrixD<Scalar> m_hat = w_state.m / (1 - std::pow(config::Optim_beta1, timestep));
        MatrixD<Scalar> v_hat = w_state.v / (1 - std::pow(config::Optim_beta2, timestep));

        // Weight update
        weights -= (config::training_learning_rate * m_hat.array() / (v_hat.array().sqrt() + config::Optim_epsilon)).matrix();

        // === BIASES ===
        VectorD<Scalar>& bias = linear->get_bias();
        const VectorD<Scalar>& grad_bias = linear->get_grad_bias();

        ParamState& b_state = bias_state[raw_layer];
        if (b_state.m.size() == 0) {
            b_state.m = VectorD<Scalar>::Zero(grad_bias.size());
            b_state.v = VectorD<Scalar>::Zero(grad_bias.size());
        }

        b_state.m = config::Optim_beta1 * b_state.m + (1 - config::Optim_beta1) * grad_bias;
        b_state.v = config::Optim_beta2 * b_state.v + (1 - config::Optim_beta2) * grad_bias.cwiseProduct(grad_bias);

        VectorD<Scalar> m_hat_b = b_state.m / (1 - std::pow(config::Optim_beta1, timestep));
        VectorD<Scalar> v_hat_b = b_state.v / (1 - std::pow(config::Optim_beta2, timestep));

        bias -= (config::training_learning_rate * m_hat_b.array() / (v_hat_b.array().sqrt() + config::Optim_epsilon)).matrix();
    }
}
