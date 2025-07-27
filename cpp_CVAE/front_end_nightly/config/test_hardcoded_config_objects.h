#pragma once
#include <memory>
#include <array>
#include <vector>
#include <random>

#include "custom_types.h"
#include "Layer_all.h"
#include "Optimizer_all.h"
#include "config_values.h"
#include "VAE.h"
#include "SequentialModule.h"
#include "param_init_utils.h"

namespace configO {

    InitFn<float> glorot = [](unsigned int in, unsigned int out, std::mt19937 gen) {
        return glorot_init<float>(in, out, gen);
    };


    // Encoder
    auto e1 = std::make_shared<SparseLinear<float>>(60000, 512, glorot);
    auto ae1 = std::make_shared<RELULayer<float>>();
    auto e2 = std::make_shared<DenseLinear<float>>(512, 256, glorot);
    auto ae2 = std::make_shared<RELULayer<float>>();
    auto e3 = std::make_shared<DenseLinear<float>>(256, 128, glorot);
    auto ae3 = std::make_shared<RELULayer<float>>();

    // Latent
    auto mu_layer = std::make_shared<DenseLinear<float>>(128, 128, glorot);
    auto logvar_layer = std::make_shared<DenseLinear<float>>(128, 128, glorot);

    // Decoder
    auto d1 = std::make_shared<SparseLinear<float>>(128, 256, glorot);
    auto ad1 = std::make_shared<RELULayer<float>>();
    auto d2 = std::make_shared<DenseLinear<float>>(256, 512, glorot);
    auto ad2 = std::make_shared<RELULayer<float>>();
    auto d3 = std::make_shared<DenseLinear<float>>(512, 60000, glorot);
    auto ad3 = std::make_shared<RELULayer<float>>();

    // Vectors
    std::vector<std::shared_ptr<Layer<float>>> encoder_layers = {e1, ae1, e2, ae2, e3, ae3};
    std::vector<std::shared_ptr<Layer<float>>> decoder_layers = {d1, ad1, d2, ad2, d3, ad3};

    auto encoder = std::make_shared<SequentialModule<float>>(encoder_layers);
    auto decoder = std::make_shared<SequentialModule<float>>(decoder_layers);
    
    
    // Model and optimizer
    std::unique_ptr<Module<float>> model = std::make_unique<VAE<float>>(encoder, decoder, mu_layer, logvar_layer);
    std::unique_ptr<Optimizer<float>> optim  = std::make_unique<Adam<float>>(0.9, 0.999, 1e-8);
}




