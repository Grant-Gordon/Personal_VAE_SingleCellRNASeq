// loss_functions.h
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include "config.h"
#include "custom_types.h"

namespace loss {

    // Mean Squared Error Loss
    template <typename Scalar>
    struct MSELoss {
        static Scalar compute(  const MatrixD<Scalar>& reconstructed,
                                const MatrixD<Scalar>& target);
    };

    //Single Sparse Row MSE Loss
    template <typename Scalar>
    struct SSRMSELoss{
        static Scalar compute(  const MatrixD<Scalar>& reconstructed,
                                const Batch<Scalar>& target); //TODO: wat?
    };

    // // Binary Cross Entropy Loss
    // template <typename Scalar>
    // struct BCELoss {
    //     static Scalar compute(  const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
    //         const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& target,
    //         Scalar epsilon = Scalar(1e-8));
    // };
        
    // // Binary Cross Entropy Loss
    // template <typename Scalar>
    // struct SSRBCELoss{
    //     static Scalar compute(  const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
    //                             const SingleSparseRow& target);
    // };

    // KL Divergence Loss for VAEs
    template <typename Scalar>
    struct KLLoss {
        static Scalar compute(  const VectorD<Scalar>& mu,
                                const VectorD<Scalar>& logvar);
    };

} // namespace loss

#include "loss_functions.tpp"