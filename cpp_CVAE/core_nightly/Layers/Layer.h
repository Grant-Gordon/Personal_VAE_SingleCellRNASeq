//Layer.h abstract class
#pragma once

#include <Eigen/SparseCore> //TODO:  include in CMAKE?
#include <Eigen/Dense>
#include <memory>
#include "custom_types.h"

template <typename Scalar>
class Layer{
    protected:
        using MatrixD = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorD = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    public: 

        //Dense X Dense
        virtual MatrixD forward(const MatrixD& input ) = 0; 

        //Input was Dense
        virtual MatrixD backward(const MatrixD& grad_output) = 0; 
        
        virtual void update_weights(Scalar learning_rate) {};
        virtual ~Layer() = default;
};