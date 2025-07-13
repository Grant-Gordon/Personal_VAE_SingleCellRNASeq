//Module.h
#pragma once

#include <vector>
#include <memory>
#include "Layer_all.h"
#include "custom_types.h"
#include "config.h"

template <typename Scalar>

class Module{
    public:
        using MatrixD = typename Layer<Scalar>::MatrixD;
        using VectorD = typename Layer<Scalar>::VectorD;

        LinearLayer<Scalar> input_layer;
        

        Module(std::vector<std::shared_ptr<Layer<Scalar>>>&& layers_vector);
        ~Module() = default;

        void add_layer(std::shared_ptr<Layer<Scalar>> layer);

        MatrixD forward(const MatrixD& std::vector<SingelSparseRow<Scalar>>& batch);

        MatrixD backward(const MatrixD upstream_grad, const std::vector<SingleSparseRow<Scalar>>& batch_input);

        void update_weights();
        
        //Getter
        Layer<Scalar>& get_layer(); 
        const Layer<Scalar>& get_layer() const; 



    private:
        std::vector<std::shared_ptr<Layer<Scalar>>> layers_vector;


};

#include "Module.tpp"