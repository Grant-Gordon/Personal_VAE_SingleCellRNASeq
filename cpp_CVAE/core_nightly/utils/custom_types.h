//custom_types.h
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
    pybind11::array_t<int> idxptr_ptr_py;

    const Scalar* vals_data;
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
        indptr_ptr_py(std::move(indtpr)),
        shape(std::move(shape)),
        nnz(nnz)
        {
            vals_ptr_data = vals_ptr_py.data();
            vols_ptr_data = cols_ptr_py.data();
            indptr_ptr_data = indptr_ptr_py.data();
        }
};
