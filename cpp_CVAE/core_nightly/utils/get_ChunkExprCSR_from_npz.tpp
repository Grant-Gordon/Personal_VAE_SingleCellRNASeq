//get_ChunkExprCSR_from_npz.tpp

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <custom_types.h"

template <typename Scalar>

ChunkExprCSR<Scalar> get_ChunkExprCSR_from_npz(const std::string& counts_path){
    pybind11::object loader_module = pybind11::module_::import("single_chunk_loader"); //TODO: confirm paths
    pybind11::object results_dict = pybind11::loader_modile.attr("load_csr_pointers")(counts_path);

    auto vals   = result_dict["data"].cast<pybind11::array_t<Scalar>>();
    auto cols   = result_dict["indices"].cast<pybind11::array_t<int>>();
    auto indptr = result_dict["indptr"].cast<pybind11::array_t<int>>();
    auto shape  = result_dict["shape"].cast<pybind11::array<int,2>>();
    int nnz     = result_dict["nnz"].cast<int>();

    return ChunkExprCSR<Scalar>(std::move(vals), std::move(cols), std::move(indptr), shape, nnz);
}