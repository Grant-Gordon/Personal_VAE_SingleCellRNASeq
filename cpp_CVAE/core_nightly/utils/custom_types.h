//custom_types.h


struct SingleSparseRow{
    const int* indices;
    const float* data;
    int nnz;
};
