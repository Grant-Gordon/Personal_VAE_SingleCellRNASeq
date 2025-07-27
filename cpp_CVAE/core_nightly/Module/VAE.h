//VAE.h
#pragma once
#include "custom_types.h"
#include "Module.h"
#include "SequentialModule.h"
#include "DenseLinear.h"

template <typename Scalar>
class VAE : public Module<Scalar>{
    public:
        VAE(
            std::shared_ptr<SequentialModule<Scalar>> encoder,
            std::shared_ptr<SequentialModule<Scalar>> decoder,
            std::shared_ptr<DenseLinear<Scalar>> mu_layer,
            std::shared_ptr<DenseLinear<Scalar>> logvar_layer
        );

        MatrixD<Scalar> forward(const MatrixD<Scalar>& input) override;
        MatrixD<Scalar> forward(const Batch<Scalar>& input) override;
        
        MatrixD<Scalar> backward(const MatrixD<Scalar>& upstream_grad) override;

        


    private:
        MatrixD<Scalar> reparameterize(const MatrixD<Scalar>& mu, const MatrixD<Scalar>& logvar);
        std::shared_ptr<DenseLinear<Scalar>> mu_layer;
        std::shared_ptr<DenseLinear<Scalar>> logvar_layer;

        std::shared_ptr<SequentialModule<Scalar>> encoder;
        std::shared_ptr<SequentialModule<Scalar>> decoder;
        MatrixD<Scalar> epsilon_cache;
        MatrixD<Scalar> mu_cache;
        MatrixD<Scalar> logvar_cache;

        const std::vector<std::shared_ptr<Layer<Scalar>>> layers_vector;
};

#include "VAE.tpp"

