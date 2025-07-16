//Adam.tpp
//AI Acknowledgement - This File utilized code from ChatGPT


#pragma once
#include <cmath>
#include <cassert>
#include "config_values.h"
#include "custom_types.h"
#include "LinearLayer.h"

template <typename Scalar>
Adam<Scalar>::Adam( int beta1, int beta2, int epsilon
): 
    timestep(0),
    beta1(beta1),
    beta2(beta2),
    epsilon(epsilon)
{
    static_assert(std::is_floating_point<Scalar>::value, "Adam: Scalar must be floating-point.");
    assert(configV::Training__lr > 0 && "Adam: learning rate must be > 0.");
    assert(this->beta1 >= 0 && this->beta1 < 1 && "Adam: configV::Optim_beta1 must be in [0, 1).");
    assert(this->beta2 >= 0 && this->beta2 < 1 && "Adam: configV::Optim_beta2 must be in [0, 1).");
    assert(this->epsilon > 0 && "Adam: configV::Optim_epsilon must be > 0.");
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

        ParamState& w_state = weight_state[layer.get()];
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
        VectorD<Scalar>& bias = linear->get_bias();
        const VectorD<Scalar>& grad_bias = linear->get_grad_bias();

        ParamState& b_state = bias_state[layer.get()];
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
