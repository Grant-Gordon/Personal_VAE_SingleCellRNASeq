#pragma once

#include <vector>
#include <memory>
#include "Layer.h"

template <typename MatrixType>

class Module{
    public:
        using Scalar = typename MatrixType::Scalar;
        Module() = default;
        ~Module() = default;

        void add_layer(std::shared_ptr<Layer<MatrixType>> layer);

        MatrixType forward(const MatrixType& input);
        MatrixType backward(const MatrixType& grad_output);
        void update_weights(Scalar learning_rate);
        
        //Getter
        std::vector<std::shared_ptr<Layer<MatrixType>>>& get_layers(); 
        const std:: vector<std::shared_ptr<Layer<MatrixType>>>& get_layers() const; 

    private:
        std::vector<std::shared_ptr<Layer<MatrixType>>> layers_vector;
        MatrixType last_input;


};

#include "Module.tpp"