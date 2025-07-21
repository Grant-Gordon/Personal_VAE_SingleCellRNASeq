//VAE.h
#pragma once
#include "custom_types.h"
#include "Module.h"
#include "SequentialModule.h"
#include "DenseLinear.h"

template <typename Scalar>
class VAE : public Module{
    public:
        VAE(SequentialModule<Scalar>& encoder,
            SequentialModule<Scalar>& decoder,
            DenseLinear<Scalar>& mu_layer,
            DenseLinear<Scalar>& logvar_layer
        );

        MatrixD<Scalar> forward(const MatrixD<Scalar>& input) override;
        MatrixD<Scalar> forward(const Batch<Scalar>& input) override;
        
        MatrixD<Sclar> backward(const MatrixD<Scalar>& upstream_grad) override;

        
    private:
        MatrixD<Scalar> remarameterize(MatrixD<Scalar> mu, MatrixD<Scalar> logvar);
        const DenseLinear<Scalar>& mu_layer;
        const DenseLinear<Scalar>& logvar_layer;

        const SequentialModule<Scalar>& encoder;
        const SequentialModule<Scalar>& decoder;
        MatrixD<Scalar>& epsilon_cache;
        MatrixD<Scalar>& mu_cache;
        MatrixD<Scalar>& logvar_cache;

};

#include "VAE.tpp"

