//Layer.h abstract class
#pragma once

#include <Eigen/Dense>
#include <memory>
#include <stdexcept> 
#include "custom_types.h"

template <typename Scalar>
class Layer{
    public: 

        //Dense->Dense Layers
        virtual MatrixD<Scalar> forward(const MatrixD<Scalar>& input) = 0; 
        virtual MatrixD<Scalar> backward(const MatrixD<Scalar>& upstream_grad) = 0; 

        //Optional Sparse batch Input
        virtual MatrixD<Scalar> forward(const Batch<Scalar>& input){
            throw std::runtime_error("Sparse input not supported");
        }

        virtual void zero_grad(){
            if (has_trainable_params()){
                throw std::runtime_error("zero_grad() not implemented for trainable layer");
            }
        }

        virtual bool supports_sparse_input() const {return false;}
        virtual bool has_trainable_params() const {return false;}
        virtual ~Layer() = default;
};