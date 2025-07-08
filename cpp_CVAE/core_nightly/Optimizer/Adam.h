//Adam.h
//AI Acknowledgement - This File utilized code from ChatGPT
#pragma once

#include "Optimizer.h" //includes "Layer.h"
#include <vector>
#include <memory>
#include <unordered_map>


template <typename Scalar>
class Adam : public Optimizer<Scalar>{
    public:

        using MatrixD = typename Layer<Scalar>::MatrixD;
        using VectorD = typename Layer<Scalar>::VectorD;

        Adam(Scalar learning_rate = 0.001,
            Scalar beta1 = 0.9,
            Scalar beta2 = 0.999,
            Scalar epsilon = 1e-8);

        void step(std::vector<std::shared_ptr<Layer<Scalar>>>& layers);

    private:
        Scalar learning_rate;
        Scalar beta1;
        Scalar beta2;
        Scalar epsilon;
        int timestep;

        struct ParamState{
            MatrixD m; // exponential moving avg of gradients
            MatrixD v; // exponential moving average of squared gradients (second moment)
        };

        std::unordered_map<Layer<Scalar>*, ParamState> weight_state;
        std::unordered_map<Layer<Scalar>*, ParamState> bias_state;


};

#include "Adam.tpp"