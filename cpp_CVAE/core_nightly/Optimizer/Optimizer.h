#pragma once

#include <memory>
#include <vector>
#include "Layer.h"

template <typename MatrixType>
class Optimizer{
    public:
        using Scalar = typename MatrixType::Scalar;
        virtual void step(std::vector<std::shared_ptr<Layer<MatrixType>>>& layers) = 0;
        virtual ~Optimizer() = default;
};