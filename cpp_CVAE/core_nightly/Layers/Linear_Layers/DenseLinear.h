
//DenseLinear.h
#pragma once

#include <Eigen/Dense>
#include "custom_types.h"
#include "TrainableLayer.h"
#include "param_init_utils.h"

template <typename Scalar>
class DenseLinear : public TrainableLayer<Scalar>{
    public:
        DenseLinear(unsigned int input_dim, unsigned int output_dim, InitFn<Scalar> init_fn);


        MatrixD<Scalar> forward(const MatrixD<Scalar>& input) override;
        MatrixD<Scalar> backward(const MatrixD<Scalar>& upstream_grad) override;

        bool supports_sparse_input() const override{return false;} //TODO: override dont even include?
        
        const MatrixD<Scalar>& get_input_cache() const override;
        MatrixD<Scalar>& get_input_cache() override;
        
        ~DenseLinear() = default;
    private:
        MatrixD<Scalar> input_cache;

};

#include "DenseLinear.tpp"
