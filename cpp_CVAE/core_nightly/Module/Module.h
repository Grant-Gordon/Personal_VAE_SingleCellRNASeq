#pragma once

#include <vector>
#include <memory>
#include "core_nightly/Layers/Layer.h"

template <typename MatrixType>

class Module{
    public:
        using Scalar = typename Matrix::Scalar;
        Module() = default;
        ~Module() = default;

        void add_layer(std::shared_ptr<Layer<MatrixType>> layer);

        MatrixType forward(const MatrixType& input);
        MatrixType backward(const MatrixType& grad_output);
        void update_weights(Scalar, learning_rate);

    private:
        std::vector<std::shared_ptr<Layer<MatrixType>> layers;
        MatrixType last_input;


};

#include "core_nightly/Module/Module.tpp"