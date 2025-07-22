
//DenseLinear.h
#pragma once

#include <Eigen/Dense>
#include "custom_types.h"
#include "Layer.h"
#include "param_init_utils.h"

template <typename Scalar>
class DenseLinear : public Layer<Scalar>{


    public:
        DenseLinear(unsigned int input_dim, unsigned int output_dim, InitFn<Scalar> init_fn);


        MatrixD<Scalar> forward(const MatrixD<Scalar>& input) override;
        MatrixD<Scalar> backward(const MatrixD<Scalar>& upstream_grad) override;

        bool supports_sparse_input() const override{return false;} //TODO: override dont even include?
        bool has_trainable_params() const  override{return true;}
        void zero_grad() override;

        const MatrixD<Scalar>& get_weights() const;
        MatrixD<Scalar>& get_weights(); 

        const MatrixD<Scalar>& get_grad_weights() const;
        
        const VectorD<Scalar>& get_bias() const;
        VectorD<Scalar>& get_bias();

        const VectorD<Scalar>& get_grad_bias() const;
        
        const MatrixD<Scalar>& get_input_cache() const;
        
        ~DenseLinear() = default;
    private:
        unsigned int input_dim;
        unsigned int output_dim;

        MatrixD<Scalar> weights;
        MatrixD<Scalar> grad_weights;

        VectorD<Scalar> bias;
        VectorD<Scalar> grad_bias;

        MatrixD<Scalar> input_cache;
};


#include "DenseLinear.tpp"
