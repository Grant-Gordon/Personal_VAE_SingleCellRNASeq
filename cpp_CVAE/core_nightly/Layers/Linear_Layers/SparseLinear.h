//SparseLinear.h
#pragma once
#include "Layer.h"
#include "TrainableLayer.h"
#include "custom_types.h"
#include "param_init_utils.h"

template <typename Scalar>
class SparseLinear : public TrainableLayer<Scalar>{
    public:
        SparseLinear(unsigned int input_dim, unsigned int output_dim, InitFn<Scalar> init_fn);

        MatrixD<Scalar> forward(const Batch<Scalar>& input) override;
        MatrixD<Scalar> backward(const MatrixD<Scalar>& upstream_grad) override;

        bool supports_sparse_input() const override  {return  true;}

        const Batch<Scalar>& get_input_cache()const;
        Batch<Scalar>& get_input_cache();

        ~SparseLinear() override = default;
    private:
        Batch<Scalar> input_cache;
};
#include "SparseLinear.tpp"