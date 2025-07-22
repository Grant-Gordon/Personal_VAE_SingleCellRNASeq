//SparseLinear.h
#pragma once
#include "Layer.h"
#include "custom_types.h"
#include "param_init_utils.h"


template <typename Scalar>
class SparseLinear : public TrainableLayer<Scalar>{
    public:

        SparseLinear(unsigned int input_dim, unsigned int output_dim, InitFn<Scalar> init_fn);

        MatrixD<Scalar> forward(const Batch<Scalar>& input) override;
        MatrixD<Scalar> backward(const MatrixD<Scalar>& upstream_grad) override;

        bool supports_sparse_input() const override  {return  true;}
        void zero_grad() override;

        ~SparseLinear() override = default;

};

#include "SparseLinear.tpp"