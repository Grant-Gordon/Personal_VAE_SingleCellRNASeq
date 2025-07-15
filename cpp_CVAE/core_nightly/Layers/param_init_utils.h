#pragma once

#include <random>
#include <cmath>
#include "config.h"
//glorot - best for sigmoiud or tanh
template<typename Scalar>
inline Scalar glorot_init(unsigned int input_dim, unsigned int output_dim, std::mt19937 gen){
    //U[-sqrt(6 / (in + out)), sqrt(6 / (in + out))]
    
    static_assert(std::is_floating_point<Scalar>::value, "glorot_init: Scalar must be floating-point.");
    assert(input_dim > 0 && output_dim > 0 && "glorot_init: input_dim and output_dim must be > 0.");


    Scalar limit = std::sqrt(6.0 / static_cast<Scalar>(input_dim + output_dim));
    std::uniform_real_distribution<Scalar> dist(-limit, limit);
    return dist(gen);
}

//HE - best for RELU leakyRELU or ELU
template<typename Scalar>
inline Scalar he_init(unsigned int input_dim, unsigned int /*output_dim*/, std::mt19937 gen){
    static_assert(std::is_floating_point<Scalar>::value, "he_init: Scalar must be floating-point.");
    assert(input_dim > 0 && "he_init: input_dim must be > 0.");


    Scalar limit = std::sqrt(6.0/ static_cast<Scalar>(input_dim));
    std::uniform_real_distribution<Scalar> dist(-limit, limit);
    return dist(gen);
}

template <typename Scalar>
inline Scalar zeros_init(unsigned int /*input_dim*/, unsigned int /*output_dim*/, std::mt19937 /*gen*/){
    static_assert(std::is_floating_point<Scalar>::value, "zeros_init: Scalar must be floating-point.");

    return Scalar(0);
}