//LinearLayer.h

#pragma once
#include <functional>
#include "Layer.h" //Incldues Eigen 
#include "param_init_utils.h"
#include "custom_types.h"
#include <optional>



template <typename Scalar>
class LinearLayer : public Layer<Scalar>{
    
    using typename Layer<Scalar>::MatrixD;
    using typename Layer<Scalar>::VectorD;

    public:

        LinearLayer(
            unsigned int input_dim, 
            unsigned int output_dim, 
            unsigned int seed, 
            InitFn init_fn);

        //Standard Dense
        MatrixD forward(const MatrixD& input) override;
        MatrixD backward(const MatrixD& grad_output) override;
        
        //Custom Sparse
        VectorD forward(const SingleSparseRow& input);
        VectorD backward(const VectorD& grad_output);
        
        void update_weights(Scalar learning_rate) override;

        //Getters
        const MatrixD& get_weights() const;
        MatrixD& get_weights(); 
        const MatrixD& get_grad_weights() const;
        
        const VectorD& get_bias() const;
        VectorD& get_bias();
        const VectorD& get_grad_bias() const;

        ~LinearLayer() override = default;

    private: 
        //TODO: point of optimization -  Could potentially infer matrices size from templates for compile time loop unrolling. 
        MatrixD weights;
        VectorD bias;      
        
        MatrixD input_cache;
        std::optional<SingleSparseRow> input_cache_sparse;

        MatrixD grad_weights;
        VectorD grad_bias;
};

#include "LinearLayer.tpp"
