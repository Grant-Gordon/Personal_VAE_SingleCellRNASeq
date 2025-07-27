//Module_mkII.h
#pragma once
#include <stdexcept>
#include <vector>
#include <memory>
#include "custom_types.h"

template <typename Scalar>
class Module{
    public:
        virtual MatrixD<Scalar> forward(const MatrixD<Scalar>& input) =0;
        virtual MatrixD<Scalar> backward(const MatrixD<Scalar>& upstream_grad) = 0;
        //Optional Sparse input support
        virtual MatrixD<Scalar> forward(const Batch<Scalar>& input){
            throw std::runtime_error("Sparse input not supported for this Module");
        }
        virtual bool supports_sparse_input() const {return false;}
        
        virtual void zero_grad(){
            for (auto& layer : this->layers_vector){
                if(layer->has_trainable_params()){
                    layer->zero_grad();
                }
            }
        }
        
        const std::vector<std::shared_ptr<Layer<Scalar>>>& get_layers_vector()const {return this->layers_vector;};

        virtual ~Module() = default;
    protected:
       const std::vector<std::shared_ptr<Layer<Scalar>>> layers_vector;
};