//Module.h
#pragma once

#include <vector>
#include <memory>
#include "Layer.h"
#include "LinearLayer.h"
#include "custom_types.h"

template <typename Scalar>

class Module{
    public:
        using MatrixD = typename Layer<Scalar>::MatrixD;
        using VectorD = typename Layer<Scalar>::VectorD;

        LinearLayer<Scalar> input_layer;
        

        Module(LinearLayer<Scalar>>&& input, std::shared_ptr<Layer<Scalar>>&& non_input_layers);
        ~Module() = default;

        void add_layer(std::shared_ptr<Layer<Scalar>> layer);

        VectorD forward_input(const SingleRowSparse& input);
        MatrixD forward(const MatrixD& dense_batch_output);

        VectorD backward_input(const VectorD& grad_output );
        MatrixD backward(const MatrixD& grad_output);

        void update_weights(Scalar learning_rate);
        
        //Getter
        std::vector<std::shared_ptr<Layer<Scalar>>>& get_non_input_layers(); 
        Layer<Scalar>& get_input_layer(); 
        const std:: vector<std::shared_ptr<Layer<Scalar>>>& get_non_input_layers() const; 
        const Layer<Scalar>& get_input_layer() const; 

        std::vector<std::shared_ptr<Layer<Scalar>>> get_all_layers();


    private:
        std::vector<std::shared_ptr<Layer<Scalar>>> layers_vector;


};

#include "Module.tpp"