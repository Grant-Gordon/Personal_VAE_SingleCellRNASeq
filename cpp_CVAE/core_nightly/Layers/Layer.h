//Layer.h abstract class
#pragma once

#include <Eigen/SparseCore> //TODO:  include in CMAKE?
#include <Eigen/Dense>
#include <memory>

template <typename MatrixType>
class Layer{
    public: 
        using Scaler = typename Matrix::Scaler;

        virtual MatrixType forward(const MatrixType& input ) = 0;          //TODO: confirm I want the outputs as T csr
        virtual MatrixType backward(const MatrixType& grad_output) = 0;    //TODO: confirm I want the outputs as T csr
        virtual void update_weights(Scalar learning_rate) {};
        virtual ~Layer() = default;
};