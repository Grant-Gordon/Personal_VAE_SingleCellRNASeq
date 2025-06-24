#pragma once

#include <random>
#include <cmath>

//glorot - best for sigmoiud or tanh
template<typename Scalar>
inline Scalar glorot_init(unsigned int input_dim, unsigned int output_dim, std::mt19937 gen){
    //U[-sqrt(6 / (in + out)), sqrt(6 / (in + out))]

    Scalar limit = std::sqrt(6.0 / static_cast<scalar>(input_dim, + output_dim));
    std::uniform_real_distribution<Scalar> dist(-limit, limit);
    return dist(gen);
}

//HE - best for RELU leakyRELU or ELU
template<typename Scalar>
inline Scalar he_init(unsigned int input_dim, unsigned int /*output_dim*/, std::mt19937 gen){
    Scalar limit = std::sqrt(6.0/ static_cast<Scalar>(input_dim));
    std::uniform_real_distribution<Scalar> dist(-limit, limit);
    return dist(gen);
}

template <typename Scalar>
inline Scalar zeros_init(unsigned int /*input_dim*/, unsigned int /*output_dim*/, std::mt19937 /*gen*/){
    return Scalar(0);
}