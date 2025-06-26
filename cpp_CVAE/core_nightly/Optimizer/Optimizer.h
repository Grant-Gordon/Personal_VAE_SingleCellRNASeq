#pragma once

#include <memory>
#include <vector>
#include "core_nightly/Layers/Layer.h"

template <typename MatrixType>
class Optimizer{
    public:
        using Scalar = typename MatrixType::Scalar;
        virtual void step(std::vector<shared_ptr<Layer<MatrixType>>>& layers) = 0;
        virtual ~Optimizer() = default;
}