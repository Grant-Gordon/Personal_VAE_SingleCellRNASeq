//TrainableLayer.h
#pragma once

#include "custom_types.h"
#include "Layer.h"

template <typename Scalar>
class TrainableLayer : public Layer<Scalar> {
    public:
        bool has_trainable_params() const  override{return true;}
        
        void zero_grad() override{
            this->grad_weights.setZero();
            this->grad_bias.setZero();
        }

        const MatrixD<Scalar>& get_weights() const{ return this->weights;}
        MatrixD<Scalar>& get_weights(){ return this->weights;}

        const MatrixD<Scalar>& get_grad_weights() const{ return this->grad_weights;}
        MatrixD<Scalar>& get_grad_weights(){ return this->grad_weights;}
        
        const VectorD<Scalar>& get_bias() const{ return this->bias;}
        VectorD<Scalar>& get_bias(){ return this->bias;}

        const VectorD<Scalar>& get_grad_bias() const{ return this->grad_bias;}
        VectorD<Scalar>& get_grad_bias(){ return this->grad_bias;}
    
        const unsigned int get_input_dim() const {return this->input_dim;}
        const unsigned int get_output_dim() const {return this->output_dim;}



    protected: 
        MatrixD<Scalar> weights;
        MatrixD<Scalar> grad_weights;

        VectorD<Scalar> bias;
        VectorD<Scalar> grad_bias;

        unsigned int input_dim;
        unsigned int output_dim;

};
