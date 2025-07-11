//Adam.h
//AI Acknowledgement - This File utilized code from ChatGPT
#pragma once

#include "Optimizer.h" //includes "Layer.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include "config.h"

template <typename Scalar>
class Adam : public Optimizer<Scalar>{
    public:

        using MatrixD = typename Layer<Scalar>::MatrixD;
        using VectorD = typename Layer<Scalar>::VectorD;

        Adam();

        void step(std::vector<std::shared_ptr<Layer<Scalar>>>& layers);

    private:
        int timestep;

        struct ParamState{
            MatrixD m; // exponential moving avg of gradients
            MatrixD v; // exponential moving average of squared gradients (second moment)
        };

        std::unordered_map<Layer<Scalar>*, ParamState> weight_state;
        std::unordered_map<Layer<Scalar>*, ParamState> bias_state;


};

#include "Adam.tpp"