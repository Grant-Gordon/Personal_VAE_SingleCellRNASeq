#pragma once

#include<Eigen/Dense>

template<typename MatrixType>
class Loss{
    public:
        using Scalar =typename MatrixType::Scalar;
        virtual Scalar forward(const MatrixType& prediction, const MatrixType& target) =0;
        virtual MatrixType backward (const MatrixType& prediction, const MatrixType& target) =0;
        virtual ~Loss() = default;

};