
//DenseLinear.h
#pragma once

#include <Eigen/Dense>
#include "custom_types.h"
#include "Layer.h"
#include "param_init_utils.h"

template <typename Scalar>
class DenseLinear : public TrainableLayer<Scalar>{
    public:
        DenseLinear(unsigned int input_dim, unsigned int output_dim, InitFn<Scalar> init_fn);


        MatrixD<Scalar> forward(const MatrixD<Scalar>& input) override;
        MatrixD<Scalar> backward(const MatrixD<Scalar>& upstream_grad) override;

        bool supports_sparse_input() const override{return false;} //TODO: override dont even include?
        void zero_grad() override;
        
        ~DenseLinear() = default;
};

#include "DenseLinear.tpp"
