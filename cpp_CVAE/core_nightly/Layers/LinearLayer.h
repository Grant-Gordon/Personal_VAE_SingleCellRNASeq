//LinearLayer.h

#pragma once
#include <functional>
#include "core_nightly/Layers/Layer.h" //Incldues Eigen 
#include "core_nightly/Layers/param_init_utils.h"

using InitFn = std::function<Scalar>(unsigned int, unsigned int, std::mt19937)>;

template <typename MatrixType>
class LinearLayer : public Layer<MatrixType>{

    public:
        using Scalar = typename MatrixType::Scalar;

        LinearLayer(unsigned int input_dim, unsigned int output_dim, unsigned int seed, InitFn init_fn);

        MatrixType forward(const MatrixType& input) override;
        MatrixType backward(const MatrixType& grad_output) override;
        void update_weights(Scalar learning_rate) override;

        ~LinearLayer() override = default;

    private: 
        //TODO: point of optimization -  Could potentially infer matrices size from templates for compile time loop unrolling. 
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> weights;
        Eigen::Matrix<Scalar, 1, Eigen::Dynamic> bias;      
        
        MatrixType input_cache;

        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> grad_weights;
        Eigen::Matrix<Scalar, 1, Eigen::Dynamic> grad_bias;
};

#include "LinearLayer.tpp"
