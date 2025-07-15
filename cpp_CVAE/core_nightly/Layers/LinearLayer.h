//LinearLayer.h

#pragma once
#include "config.h"
#include <functional>
#include <optional>
#include "Layer.h" //Incldues Eigen 
#include "custom_types.h"
#include "param_init_utils.h"
#include "custom_types.h"


template <typename Scalar>
class LinearLayer : public Layer<Scalar>{
    public:

        //TODO: can pass input, output dims, and init fn with config 
        LinearLayer(
            unsigned int input_dim, 
            unsigned int output_dim, 
            InitFn<Scalar> init_fn
            );

        //Standard Dense
        MatrixD<Scalar> forward(const MatrixD<Scalar>& input) override;
        MatrixD<Scalar> backward(const MatrixD<Scalar>& grad_output) override;
        
        //Custom Sparse
        VectorD<Scalar> forward(const SingleSparseRow<Scalar>& input);
        VectorD<Scalar> backward(const VectorD<Scalar>& upstream_grad, const SingleSparseRow<Scalar>& input);
        
        void update_weights(Scalar learning_rate) override;

        //Getters
        bool has_trainable_params() const  override {return true;}

        const MatrixD<Scalar>& get_weights() const;
        MatrixD<Scalar>& get_weights(); 
        const MatrixD<Scalar>& get_grad_weights() const;
        
        const VectorD<Scalar>& get_bias() const;
        VectorD<Scalar>& get_bias();
        const VectorD<Scalar>& get_grad_bias() const;

        ~LinearLayer() override = default;

    private: 
        //TODO: point of optimization -  Could potentially infer matrices size from templates for compile time loop unrolling. 
        MatrixD<Scalar> weights;
        VectorD<Scalar> bias;      
        
        MatrixD<Scalar> input_cache;

        MatrixD<Scalar> grad_weights;
        VectorD<Scalar> grad_bias;
        //MatrixD<Scalar> grad_inputs;//TODO: maybe not really necessary
};

#include "LinearLayer.tpp"
