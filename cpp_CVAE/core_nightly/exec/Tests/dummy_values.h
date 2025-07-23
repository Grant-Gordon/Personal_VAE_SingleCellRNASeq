// dummy_values.h
// AI Acknowledgement: This file was generated using  ChatGPT

#pragma once
#include "custom_types.h"
#include <vector>
#include <memory>

// === Dummy Dense Input ===
// Shape: batch_size=4, input_dim=5
template <typename Scalar>
inline MatrixD<Scalar> get_dummy_input_matrix() {
    MatrixD<Scalar> mat(4, 5);
    mat << 1.0,  0.0, -1.0, 2.0,  0.5,
           0.5,  2.0,  0.3, 0.0, -0.7,
          -1.5,  1.2,  0.0, 0.8,  1.1,
           0.0, -0.4,  0.9, 1.3, -1.0;
    return mat;
}

// === Dummy Upstream Gradient ===
// Shape: batch_size=4, output_dim=6
template <typename Scalar>
inline MatrixD<Scalar> get_dummy_upstream_grad(int output_dim = 6) {
    MatrixD<Scalar> grad(4, output_dim);
    grad << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
            0.2, 0.1, 0.4, 0.3, 0.6, 0.5,
            0.3, 0.4, 0.1, 0.2, 0.5, 0.6,
            0.4, 0.3, 0.2, 0.1, 0.6, 0.5;
    return grad;
}

// === Helper Struct for Dummy Sparse Rows ===
template <typename Scalar>
struct TestableSparseRow {
    std::vector<int> indices;
    std::vector<Scalar> data;
    int nnz;

    operator SingleSparseRow<Scalar>() const {
        return {
            indices.data(),
            data.data(),
            nnz
        };
    }
};

// === Dummy Sparse Input ===
// Shape: batch_size=4, input_dim=5
template <typename Scalar>
inline Batch<Scalar> get_dummy_sparse_batch() {
    std::vector<TestableSparseRow<Scalar>> temp_rows(4);

    temp_rows[0] = {{0, 2, 4}, {1.0, -1.0, 0.5}, 3};
    temp_rows[1] = {{1, 3},    {0.5, 2.0},       2};
    temp_rows[2] = {{0, 1, 4}, {-1.5, 1.2, 1.1}, 3};
    temp_rows[3] = {{2, 3},    {0.9, 1.3},       2};

    Batch<Scalar> batch;
    batch.reserve(4);
    for (const auto& row : temp_rows) {
        batch.emplace_back(std::make_unique<SingleSparseRow<Scalar>>(row));
    }

    return batch;
}

// === Dummy Initializer ===
// Deterministic function for test reproducibility
template <typename Scalar>
inline InitFn<Scalar> get_dummy_init_fn() {
    return [](unsigned int in_dim, unsigned int out_dim, std::mt19937&) {
        return 0.05 * (in_dim + out_dim);  // Simple deterministic value
    };
}

