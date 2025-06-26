#pragma once
#include "Layer.h"
#include <Eigen/Dense>


template <typename MatrixType>
class RELULayer: public Layer<MatrixType>{
    public:
        using Scalar = typename MatrixType::Scalar;
        RELULayer() = default;
        ~RELULayer() override = default;

        MatrixType forward(const MatrixType& input) override; //TODO: point of optimization - may be able to overload forward and pacward to pass rvals using move semantics to avoid copies
        MatrixType backward(const MatrixType& grad_output) override;

    private:
        MatrixType input_cache;

        

};

#include "RELU.tpp"