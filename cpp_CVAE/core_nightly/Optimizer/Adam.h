//Adam.h
//AI Acknowledgement - This File utilized code from ChatGPT
#pragma once

#include "Optimizer.h" //includes "Layer.h"
#include <vector>
#include <memory>
#include <unordered_map>


template <typename MatrixType>
class Adam : public Optimizer<MatrixType>{
    public:
        using Scalar = typename MatrixType::Scalar;

        Adam(Scalar learning_rate = 0.001,
            Scalar beta1 = 0.9,
            Scalar beta2 = 0.999,
            Scalar epsilon = 1e-8);

        void step(std::vector<std::shared_ptr<Layer<MatrixType>>>& layer);

    private:
        Scalar learning_rate;
        Scalar beta1;
        Scalar beta2;
        Scalar epsilon;
        int timestep;

        struct ParamState{
            MatrixType m; // exponential moving avg of gradients
            MatrixType v; // exponential moving average of squared gradients (second moment)
        };

        std::unordered_map<Layer<MatrixType>*, ParamState> weight_state;
        std::unordered_map<Layer<MatrixType>*, ParamState> bias_state;


};

#include "Adam.tpp"