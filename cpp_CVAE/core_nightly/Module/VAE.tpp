//VAE.tpp
#pragma once

#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include "custom_types.h"
#include "Module.h"
#include "DenseLinear.h"
#include "SequentialModule.h"

template <typename Scalar>
VAE<Scalar>::VAE(
    SequentialModule<Scalar>& encoder,
    SequentialModule<Scalar>& decoder,
    DenseLinear<Scalar>& mu_layer,
    DenseLinear<Scalar>& logvar_layer
){   

    this->encode = encoder;
    this->decoder = decoder;
    this->mu_layer = mu_layer;
    this->logvar_layer = logvar_layer;

    //Add all layers to layer vector;
    this->layers_vector.reserve(this->encoder->get_layers_vector().size() + this->decoder->get_layers_vector().size() + 2);
    this->layers_vector.insert(this->layers_vector.end(), this->encoder->get_layers_vector().begin(), this->encoder->get_layers_vector().end());
    this->layers_vector.push_back(this->mu_layer);
    this->layers_vector.push_back(this->logvar_layer);
    this->layers_vector.insert(this->layers_vector.end(), this->decoder->get_layers_vector().begin(), this->decoder->get_layers_vector().end());
}

//Dense input
template <typename Scalar>
MatrixD<Scalar> VAE<Scalar>::forward(const MatrixD<Scalar>& input){
    MatrixD<Scalar> encoded = this->encoder->forward(input);
    this->mu_cache = this->mu_layer->forward(encoded);
    this->logvar_cache = this->logvar_layer->forward(encoded);
    MatrixD<Scalar> z = this->reparameterize(this->mu_cache, this->logvar_cache);
    MatrixD<Scalar> decoded = this->decoder->forward(z);
    return decoded;
}

//Sparse Input
template <typename Scalar>
MatrixD<Scalar> VAE<Scalar>::forward(const Batch<Scalar>& input){
    MatrixD<Scalar> encoded = this->encoder->forward(input);
    this->mu_cache = this->mu_layer->forward(encoded);
    this->logvar_cache = this->logvar_layer->forward(encoded);
    MatrixD<Scalar> z = this->reparameterize(this->mu_cache, this->logvar_cache);
    MatrixD<Scalar> decoded = this->decoder->forward(z);
    return decoded;
}


template <typename Scalar>
MatrixD<Scalar> VAE<Scalar>::reparameterize(const MatrixD<Scalar>& mu, const MatrixD<Scalar>& logvar){
    MatrixD<Scalar> std = (0.5 * logvar).array().exp().matrix();// logvar = log(std^2)
    this->epsilon_cache = MatrixD<Scalar>::Random(mu.rows(), mu.cols());//TODO: This might throw a bug
    return mu + (this->epsilon_cache * std);
}

template <typename Scalar>
MatrixD<Scalar> VAE<Scalar>::backward(const MatrixD<Scalar>& upstream_grad){
    //recompute std from logvar
    MatrixD<Scalar> std = (0.5 * this->logvar_cache.array()).exp().matrix();
    //1)backprop through decoder with Recon loss
    MatrixD<Scalar> dL_dz = this->decoder->backward(upstream_grad);
    
    //2)backward through the Reparameterization trick + KL loss 
    //dL/d_mu
    MatrixD<Scalar> dL_dmu = dL_dz + this->mu_cache; //grads from recon grad + KL

    //dL/d_logvar
    MatrixD<Scalar> dL_dlogvar = (0.5 * dL_dz.array() * this->epsilon_cache.array() * std.array()).matrix(); //grad from recon loss
    dL_dlogvar += 0.5 * (std.array().square() -1).matrix(); //grad from KL loss

    //3)backprop into mu and logvar Layers
    MatrixD<Scalar> mu_layer_downstream_grad = this->mu_layer.backward(dL_dmu);
    MatrixD<Scalar> logvar_layer_downstream_grad = this->logvar_layer.backward(dL_dlogvar);

    //4)backprop through encoder
    return this->encoder.backward(mu_layer_downstream_grad + logvar_layer_downstream_grad); //Add them because of some multivariate calculus chain rule stuff. Idk, 
}

