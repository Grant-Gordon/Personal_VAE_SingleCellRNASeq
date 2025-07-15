//RELU.h
#pragma once

#include <Eigen/Dense>
#include "Layer.h"
#include "custom_types.h"


template <typename Scalar>
class RELULayer: public Layer<Scalar>{
    public:
        RELULayer() = default;
        ~RELULayer() override = default;

        MatrixD forward(const MatrixD& input) override; //TODO: point of optimization - may be able to overload forward and pacward to pass rvals using move semantics to avoid copies
        MatrixD backward(const <MatrixD>& grad_output) override;

    private:
        MatrixD input_cache;

        

};

#include "RELU.tpp"