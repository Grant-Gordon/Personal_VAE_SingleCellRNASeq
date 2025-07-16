// loss_functions.tpp
#pragma once
#include "config_values.h"
#include "custom_types.h"
namespace loss {

   
    // MSE
    template <typename Scalar>
    Scalar MSELoss<Scalar>::compute(const MatrixD<Scalar>& reconstructed, const MatrixD<Scalar>& target) {
        return (reconstructed - target).squaredNorm() / static_cast<Scalar>(reconstructed.rows());
    }

    template <typename Scalar>
    MatrixD<Scalar> MSELoss<Scalar>::gradients(const MatrixD<Scalar>& reconstructed, const MatrixD<Scalar>& target){
        return (reconstructed - target) / static_cast<Scalar>(2.0 / reconstructed.rows());
    }

    //SSR MSE
    template <typename Scalar>
    Scalar SSRMSELoss<Scalar>::compute( const MatrixD<Scalar>& reconstructed, const Batch<Scalar>& targets) {
        const int feature_size = reconstructed.cols();
        int batch_size = targets.size();
        Scalar total_error = 0;
        
        #pragma omp parallel for reduction(+:total_error)
       for(int i = 0; i < batch_size; ++i) {
            const SingleSparseRow<Scalar>& target_row = *targets[i];
            const auto& reconstructed_row = reconstructed.row(i);

            for (int k = 0; k < target_row.nnz; ++k){
                int col = target_row.indices[k];
                Scalar val = target_row.data[k]; 
                Scalar difference = reconstructed_row(col) - val;
                total_error += difference * difference;
            }
        }

        return total_error / configV::Training__batch_size;
    }

    template <typename Scalar>
    MatrixD<Scalar> SSRMSELoss<Scalar>::gradients(const MatrixD<Scalar>& reconstructed, const Batch<Scalar>& target){
        int batch_size = target.size();
        int output_dim = reconstructed.cols();
        MatrixD<Scalar> gradients(batch_size, output_dim); 
        gradients.setZero();

        for(size_t i=0; i < batch_size; ++i){ // For ALl samples
            const SingleSparseRow<Scalar>& row = *target[i];
            int nnz = row.nnz;
            for(size_t j = 0; j < nnz; ++j){ //For all nnz in input (target) sample
                int col = row.indices[j]; 
                Scalar val = row.data[j];
                Scalar diff = reconstructed(i, col) -val;
                gradients(i, col) = (2.0* diff) / (batch_size * nnz);
            }
        }
        return gradients;
    }



    // // BCE
    // template <typename Scalar>
    // Scalar BCELoss<Scalar>::compute(const MatrixD<Scalar>& input,
    //                                 const MatrixD<Scalar>& target,
    //                                 Scalar epsilon) {
    //     auto clipped = input.array().min(1 - epsilon).max(epsilon);
    //     return -((target.array() * clipped.log()) + ((1 - target.array()) * (1 - clipped).log())).sum() / static_cast<Scalar>(input.rows());
    // }

    // KL Divergence
    template <typename Scalar>
    Scalar KLLoss<Scalar>::compute(const VectorD<Scalar>& mu, const VectorD<Scalar>& logvar) {
        return static_cast<Scalar>(-0.5) * (1 + logvar.array() - mu.array().square() - logvar.array().exp()).sum();
    }

} // namespace loss
