#pragma once

#include "core_nightly/Layers/Layer.h"


template <typename MatrixType>
class SigmoidLayer : public Layer<MatrixType>{
    public:
        using Scalar = typename MatrixType::Scalar;

        SigmoidLayer() = default;
        ~SigmoidLayer() override = default;

        MatrixType forward(const MatrixType& input) override;
        MatrixType backward(const MatrixType& grad_output) override;

    private:
        MatrixType output_cache; //Store output from forward to be used in backward
};

#include "core_nightly/Layers/Activation_Layers/sigmoid.tpp"