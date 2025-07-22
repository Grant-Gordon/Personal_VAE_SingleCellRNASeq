//SparseLinear.h
#pragma once
#include "Layer.h"
#include "custom_types.h"
#include "param_init_utils.h"


template <typename Scalar>
class SparseLinear : public Layer<Scalar>{
    public:

        SparseLinear(unsigned int input_dim, unsigned int output_dim, InitFn<Scalar> init_fn);

        MatrixD<Scalar> forward(const Batch<Scalar>& input) override;
        MatrixD<Scalar> backward(const MatrixD<Scalar>& upstream_grad) override;

        bool supports_sparse_input() const override  {return  true;}
        bool has_trainable_params() const override {return true;}
        void zero_grad() override;


        //Getters
        const MatrixD<Scalar>& get_weights() const;
        MatrixD<Scalar>& get_weights(); 

        const MatrixD<Scalar>& get_grad_weights() const;
        
        const VectorD<Scalar>& get_bias() const;
        VectorD<Scalar>& get_bias();

        const VectorD<Scalar>& get_grad_bias() const;
        
        const Batch<Scalar>& get_input_cache() const;
       
        ~SparseLinear() override = default;

    private:

        unsigned int input_dim;
        unsigned int output_dim;

        MatrixD<Scalar> weights;
        MatrixD<Scalar> weights_grad;
        
        VectorD<Scalar> bias;
        VectorD<Scalar> bias_grad;
        
        Batch<Scalar> input_cache;
};

#include "SparseLinear.tpp"