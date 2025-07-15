//get_ChunkExprCSR_from_npz.tpp
#pragma once

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "custom_types.h"

template <typename Scalar>

ChunkExprCSR<Scalar> get_ChunkExprCSR_from_npz(const std::string& counts_path){
    pybind11::object loader_module = pybind11::module_::import("single_chunk_loader"); //TODO: confirm paths
    pybind11::object results_dict = loader_module.attr("load_csr_pointers")(counts_path);

    auto vals   = results_dict["data"].cast<pybind11::array_t<Scalar>>();
    auto cols   = results_dict["indices"].cast<pybind11::array_t<int>>();
    auto indptr = results_dict["indptr"].cast<pybind11::array_t<int>>();
    int nnz     = results_dict["nnz"].cast<int>();
    pybind::array shape_array = results_dict["shape"].cast<pybind11:array>();
    auto shape_buffer = shape_array.request(); //pybind11::buffer_info
    std::array<int,2> shape =  {
        static_cast<int>(shape_buffer.shape[0]),
        static_cast<int>(shape_buffer.shape[1])
    };

    return ChunkExprCSR<Scalar>(std::move(vals), std::move(cols), std::move(indptr), shape, nnz);
}