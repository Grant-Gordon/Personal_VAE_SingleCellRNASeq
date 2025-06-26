#pragma once
#include "Loss.h"

template <typename MatrixType>
class MSE : public Loss<MatrixType>{
    public: 
        using Scalar = typename MatrixType::Scalar;

        Scalar forward(const MatrixType& predicted, const MatrixType& target) override;
        MatrixType backward(const MatrixType& predicted, const MatrixType& target) override;

        ~MSE() override = default;
};  

#include "MSE.tpp"