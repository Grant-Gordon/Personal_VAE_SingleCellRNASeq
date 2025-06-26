//LinearLayer.h

#pragma once
#include <functional>
#include "Layer.h" //Incldues Eigen 
#include "param_init_utils.h"

template <typename MatrixType>
class LinearLayer : public Layer<MatrixType>{

    public:
        using Scalar = typename MatrixType::Scalar;
        using InitFn = std::function<Scalar(unsigned int, unsigned int, std::mt19937)>;


        LinearLayer(
            unsigned int input_dim, 
            unsigned int output_dim, 
            unsigned int seed, 
            InitFn init_fn);

        MatrixType forward(const MatrixType& input) override;
        MatrixType backward(const MatrixType& grad_output) override;
        void update_weights(Scalar learning_rate) override;

        //Getters
        const MatrixType& get_weights() const;
        MatrixType& get_weights(); 
        const MatrixType& get_grad_weights() const;
        
        const MatrixType& get_bias() const;
        MatrixType& get_bias();
        const MatrixType& get_grad_bias() const;

        ~LinearLayer() override = default;

    private: 
        //TODO: point of optimization -  Could potentially infer matrices size from templates for compile time loop unrolling. 
        MatrixType weights;
        MatrixType bias;      
        
        MatrixType input_cache;

        MatrixType grad_weights;
        MatrixType grad_bias;
};

#include "LinearLayer.tpp"
