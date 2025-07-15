//Layer.h abstract class
#pragma once

#include <Eigen/Dense>
#include <memory>
#include "custom_types.h"

template <typename Scalar>
class Layer{
    public: 

        //Dense X Dense
        virtual MatrixD<Scalar> forward(const MatrixD<Scalar>& input ) = 0; 

        //Input was Dense
        virtual MatrixD<Scalar> backward(const MatrixD<Scalar>& grad_output) = 0; 
        
        virtual void update_weights(Scalar learning_rate) {};

        virtual bool has_trainable_params() const {return false;}
        virtual ~Layer() = default;
};