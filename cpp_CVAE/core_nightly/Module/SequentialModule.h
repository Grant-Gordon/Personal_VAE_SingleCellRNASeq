//SequentialModuel.h
#pragma once
#include <vector>
#include "custom_types.h"
#include "Module_mkII.h"//TODO:rename to actual Module 
#include "Layer.h"

template <typename Scalar>
class SequentialModule : public Module<Scalar>{
    public:
        SequentialModule(std::vector<std::shared_ptr<Layer<Scalar>>>&& layers_vector);
        
        MatrixD<Scalar> forward(const MatrixD<Scalar>& input) override;
        MatrixD<Scalar> backward(const MatrixD<Scalar>& upstream_grad) override;
        //Optional Sparse input support
        MatrixD<Scalar> forward(const Batch<Scalar>& input) override;
        
        bool supports_sparse_input() const override;
        void zero_grad()override;

    private:
        const std::vector<std::shared_ptr<Layer<Scalar>>> layers_vector;    
};