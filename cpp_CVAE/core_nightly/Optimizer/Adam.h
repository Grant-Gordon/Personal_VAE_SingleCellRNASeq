//Adam.h
//AI Acknowledgement - This File utilized code from ChatGPT
#pragma once

#include "Optimizer.h" //includes "Layer.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include "config_values.h"
#include "custom_types.h"

template <typename Scalar>
class Adam : public Optimizer<Scalar>{
    public:

        Adam(int beta1, int beta2, int epsilon);

        void step(std::vector<std::shared_ptr<Layer<Scalar>>>& layers);

    private:
        int timestep;
        int beta1;
        int beta2;
        int epsilon; 
        struct ParamState{
            MatrixD<Scalar> m; // exponential moving avg of gradients
            MatrixD<Scalar> v; // exponential moving average of squared gradients (second moment)
        };

        std::unordered_map<Layer<Scalar>*, ParamState> weight_state; 
        std::unordered_map<Layer<Scalar>*, ParamState> bias_state;


};

#include "Adam.tpp"