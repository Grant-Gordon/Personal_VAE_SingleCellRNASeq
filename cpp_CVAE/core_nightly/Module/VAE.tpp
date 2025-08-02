//VAE.tpp
#pragma once

#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include "custom_types.h"
#include "Module.h"
#include "DenseLinear.h"
#include "SequentialModule.h"
#include "macros.h"

template <typename Scalar>
VAE<Scalar>::VAE(
    std::shared_ptr<SequentialModule<Scalar>> encoder,
    std::shared_ptr<SequentialModule<Scalar>> decoder,
    std::shared_ptr<DenseLinear<Scalar>> mu_layer,
    std::shared_ptr<DenseLinear<Scalar>> logvar_layer
):
  encoder(encoder),
  decoder(decoder),
  mu_layer(mu_layer),
  logvar_layer(logvar_layer),
  layers_vector([&]() {
      std::vector<std::shared_ptr<Layer<Scalar>>> vec;
      vec.reserve(encoder->get_layers_vector().size() + decoder->get_layers_vector().size() + 2);
      vec.insert(vec.end(), encoder->get_layers_vector().begin(), encoder->get_layers_vector().end());
      vec.push_back(mu_layer);
      vec.push_back(logvar_layer);
      vec.insert(vec.end(), decoder->get_layers_vector().begin(), decoder->get_layers_vector().end());
      return vec;
  }())

{}

//Dense input
template <typename Scalar>
MatrixD<Scalar> VAE<Scalar>::forward(const MatrixD<Scalar>& input){
    ASSERT(input.rows() > 0 && input.cols() > 0);
    MatrixD<Scalar> encoded = this->encoder->forward(input);
    DASSERT(encoded.rows() == input.rows());


    this->mu_cache = this->mu_layer->forward(encoded);
    this->logvar_cache = this->logvar_layer->forward(encoded);
    //Validate cache shapes
    DASSERT(mu_cache.rows() == encoded.rows() && mu_cache.cols() > 0);
    DASSERT(logvar_cache.rows() == encoded.rows());
    DASSERT(mu_cache.rows() == logvar_cache.rows());
    DASSERT(mu_cache.cols() == logvar_cache.cols());

    MatrixD<Scalar> z = this->reparameterize(this->mu_cache, this->logvar_cache);
    //Validate reparameterized shapes
    DASSERT(z.rows() == mu_cache.rows());
    DASSERT(z.cols() == mu_cache.cols());

    MatrixD<Scalar> decoded = this->decoder->forward(z);
    DASSERT(decoded.rows() == input.rows());

    return decoded;
}

//Sparse Input
template <typename Scalar>
MatrixD<Scalar> VAE<Scalar>::forward(const Batch<Scalar>& input){
    ASSERT(!input.empty());
    MatrixD<Scalar> encoded = this->encoder->forward(input);
    DASSERT(encoded.rows() == input.size());


    this->mu_cache = this->mu_layer->forward(encoded);
    this->logvar_cache = this->logvar_layer->forward(encoded);
    //Validate cache shapes
    DASSERT(mu_cache.rows() == encoded.rows() && mu_cache.cols() > 0);
    DASSERT(logvar_cache.rows() == encoded.rows());
    DASSERT(mu_cache.rows() == logvar_cache.rows());
    DASSERT(mu_cache.cols() == logvar_cache.cols());

    MatrixD<Scalar> z = this->reparameterize(this->mu_cache, this->logvar_cache);
    //Validate reparameterized shapes
    DASSERT(z.rows() == mu_cache.rows());
    DASSERT(z.cols() == mu_cache.cols());

    MatrixD<Scalar> decoded = this->decoder->forward(z);
    DASSERT(decoded.rows() == input.size());

    return decoded;
}


template <typename Scalar>
MatrixD<Scalar> VAE<Scalar>::reparameterize(const MatrixD<Scalar>& mu, const MatrixD<Scalar>& logvar){
    ASSERT(mu.rows() == logvar.rows() && mu.cols() == logvar.cols());
    ASSERT(mu.rows() > 0 && mu.cols() > 0);

    MatrixD<Scalar> std = (0.5 * logvar.array()).exp().matrix();
    this->epsilon_cache = MatrixD<Scalar>::Random(mu.rows(), mu.cols());
    return mu + this->epsilon_cache.cwiseProduct(std);
}

template <typename Scalar>
MatrixD<Scalar> VAE<Scalar>::backward(const MatrixD<Scalar>& upstream_grad){
    DASSERT(upstream_grad.rows() > 0 && upstream_grad.cols() > 0);
    
    //recompute std from logvar
    MatrixD<Scalar> std = (0.5 * this->logvar_cache.array()).exp().matrix();
    
    //1)backprop through decoder with Recon loss
    MatrixD<Scalar> dL_dz = this->decoder->backward(upstream_grad);
    DASSERT(dL_dz.rows() == this->mu_cache.rows() && dL_dz.cols() == this->mu_cache.cols());

    //2)backward through the Reparameterization trick + KL loss 
    //dL/d_mu
    MatrixD<Scalar> dL_dmu = dL_dz + this->mu_cache; //grads from recon grad + KL

    //dL/d_logvar
    MatrixD<Scalar> dL_dlogvar = (0.5 * dL_dz.array() * this->epsilon_cache.array() * std.array()).matrix(); //grad from recon loss
    dL_dlogvar += 0.5 * (std.array().square() -1).matrix(); //grad from KL loss

    //3)backprop into mu and logvar Layers
    MatrixD<Scalar> mu_layer_downstream_grad = this->mu_layer->backward(dL_dmu);
    MatrixD<Scalar> logvar_layer_downstream_grad = this->logvar_layer->backward(dL_dlogvar);

    //4)backprop through encoder
    return this->encoder->backward(mu_layer_downstream_grad + logvar_layer_downstream_grad); //Add them because of some multivariate calculus chain rule stuff. Idk, 
}

