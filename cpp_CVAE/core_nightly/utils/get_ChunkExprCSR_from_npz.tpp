//get_ChunkExprCSR_from_npz.tpp
#pragma once

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "custom_types.h"
#include "macros.h"

template <typename Scalar>

ChunkExprCSR<Scalar> get_ChunkExprCSR_from_npz(const std::string& counts_path){
    VERBOSEL2("Inside utils/get_chunkEXPRCSR_from_npz");
    pybind11::object loader_module = pybind11::module_::import("single_chunk_loader"); //TODO: confirm paths
    ASSERT(pybind11::hasattr(loader_module, "load_csr_pointers"));
    if (!pybind11::hasattr(loader_module, "load_csr_pointers")) {
        std::cerr << "[ERROR] Module 'single_chunk_loader' has no attribute 'load_csr_pointers'\n";
        std::abort();
    }
    pybind11::object results_dict = loader_module.attr("load_csr_pointers")(counts_path);
    VERBOSEL2("Retrieved 'results_dict' from loader_module");
    auto vals   = results_dict["data"].cast<pybind11::array_t<Scalar>>();
    auto cols   = results_dict["indices"].cast<pybind11::array_t<int>>();
    auto indptr = results_dict["indptr"].cast<pybind11::array_t<int>>();
    int nnz     = results_dict["nnz"].cast<int>();
    pybind11::array shape_array = results_dict["shape"].cast<pybind11::array>();
    auto shape_buffer = shape_array.request(); //pybind11::buffer_info
    auto shape_data = static_cast<int*>(shape_buffer.ptr);
    std::array<int,2> shape =  {
        shape_data[0],
        shape_data[1]
    };
    ASSERT(shape[0] > 0 && shape[1] > 0);
    VERBOSEL2("Returning ChunkExprCSR from get_ChunkExprCSR_from_npz");
    return ChunkExprCSR<Scalar>(std::move(vals), std::move(cols), std::move(indptr), shape, nnz);
}