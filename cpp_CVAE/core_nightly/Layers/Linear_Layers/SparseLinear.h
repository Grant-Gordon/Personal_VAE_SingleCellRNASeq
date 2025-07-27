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
        MatrixD<Scalar> forward(const MatrixD<Scalar>& input) override{ throw std::runtime_error("SparseLinear::forward(MatrixD) is not supported. Use the sparse Batch input version.");}

        MatrixD<Scalar> backward(const MatrixD<Scalar>& upstream_grad) override;

        bool supports_sparse_input() const override  {return  true;}

        const Batch<Scalar>& get_input_cache()const; //TODO: confirm getters for new ptr input_cache
        const Batch<Scalar>* get_input_cache_ptr() const;
        Batch<Scalar>& get_input_cache();

        ~SparseLinear() override = default;
    private:
        const Batch<Scalar>* input_cache_ptr; //TODO const pointers are confusing? think this allows me to reassgn for each new batch??
};
#include "SparseLinear.tpp"