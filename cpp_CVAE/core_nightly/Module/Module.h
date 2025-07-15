//Module.h
#pragma once

#include <vector>
#include <memory>
#include "config.h"
#include "custom_types.h"
#include "Layer_all.h"

template <typename Scalar>

class Module{
    public:
        LinearLayer<Scalar> input_layer;
        

        Module(std::vector<std::shared_ptr<Layer<Scalar>>>&& layers_vector);
        ~Module() = default;

        void add_layer(std::shared_ptr<Layer<Scalar>> layer);

        MatrixD<Scalar> forward(const Batch<Scalar>& batch); //TDOO: these will break if I ever have a module/head that does not take in sparse. Need to overlaod

        MatrixD<Scalar> backward(const MatrixD<Scalar> upstream_grad, const Batch<Scalar>& batch_input);

        void update_weights();
        
        //Getter
        std::vector<std::shared_ptr<Layer<Scalar>>>& get_layers(); 
        const std::vector<std::shared_ptr<Layer<Scalar>>>& get_layers() const; 



    private:
        std::vector<std::shared_ptr<Layer<Scalar>>> layers_vector;


};

#include "Module.tpp"