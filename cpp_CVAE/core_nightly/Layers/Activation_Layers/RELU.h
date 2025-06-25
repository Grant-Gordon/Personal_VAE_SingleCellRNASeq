#pragma once
#include "core_nightly/Layers/Layer.h"
#include <Eigen/Dense>


template <typename MatrixType>
class RELULayer: public Layer<MatrixType>{
    public:
        using Sclar = typename MatrixType::Scalar;
        RELU() = default;
        ~RELU() override = defualt;

        MatrixType forward(const MatrixType& input) override; //TODO: point of optimization - may be able to overload forward and pacward to pass rvals using move semantics to avoid copies
        MatrixType backward(const MatrixType& grad_output) override;

    private:
        MatrixType input_cache;

        

};

#include "core_nightly/Layers/Activation_Layers/RELU.tpp"