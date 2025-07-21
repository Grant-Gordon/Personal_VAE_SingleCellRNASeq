//Module_mkII.h
#pragma once
#include <stdexcept>
#include "custom_types.h"

template <typename Scalar>
class Module{
    public:
        virtual MatrixD<Scalar> forward(const MatrixD<Scalar>& input) =0;
        virtual MatrixD<Scalar> backward(const MatrixD<Scalar>& upstream_grad) = 0;
        //Optional Sparse input support
        virtual MatrixD<Scalar> forward(const Batch<Scalar>& input){
            throw std::runtime_error("Sparse input not supported for this Module");
        }
        virtual bool supports_sparse_input() const {return false;}
        virtual void zero_grad() = 0;

        virtual ~Module() = default;
};