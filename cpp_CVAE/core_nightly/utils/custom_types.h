//custom_types.h
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include <Eigen/Dense>

template <typename Scalar>
using MatrixD = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Scalar>
using VectorD = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;


template <typename Scalar>
struct SingleSparseRow{
    const int* indices;
    const Scalar* data;
    int nnz;
};

template <typename Scalar>
struct ChunkExprCSR{
    pybind11::array_t<Scalar> vals_ptr_py;
    pybind11::array_t<int> cols_ptr_py;
    pybind11::array_t<int> indptr_ptr_py;

    const Scalar* vals_ptr_data;
    const int* cols_ptr_data;
    const int* indptr_ptr_data;
    
    std::array<int, 2> shape; 
    int nnz; 
    
    ChunkExprCSR(
        pybind11::array_t<Scalar> vals,
        pybind11::array_t<int> cols,
        pybind11::array_t<int> indptr,
        std::array<int,2> shape,
        int nnz
    ):
    vals_ptr_py(std::move(vals)),
    cols_ptr_py(std::move(cols)),
    indptr_ptr_py(std::move(indptr)),
    shape(std::move(shape)),
    nnz(nnz)
    {
        vals_ptr_data = vals_ptr_py.data();
        cols_ptr_data = cols_ptr_py.data();
        indptr_ptr_data = indptr_ptr_py.data();
    }
};


template <typename Scalar>
using Batch = typename std::vector<std::unique_ptr<SingleSparseRow<Scalar>>>;

template <typename Scalar>
using InitFn = std::function<Scalar(unsigned int, unsigned int, std::mt19937)>;