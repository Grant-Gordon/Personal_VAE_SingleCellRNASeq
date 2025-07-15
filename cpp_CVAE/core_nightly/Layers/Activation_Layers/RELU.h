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

        MatrixD<Scalar> forward(const MatrixD<Scalar>& input) override; //TODO: point of optimization - may be able to overload forward and pacward to pass rvals using move semantics to avoid copies
        MatrixD<Scalar> backward(const MatrixD<Scalar>& grad_output) override;

    private:
        MatrixD<Scalar> input_cache;

        

};

#include "RELU.tpp"