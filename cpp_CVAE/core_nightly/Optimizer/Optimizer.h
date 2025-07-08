#pragma once

#include <memory>
#include <vector>
#include "Layer.h"

template <typename Scalar>
class Optimizer{
    public:
    
        virtual void step(std::vector<std::shared_ptr<Layer<Scalar>>>& layers) = 0;
        virtual ~Optimizer() = default;
};