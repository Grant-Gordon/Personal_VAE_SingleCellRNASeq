// loss_functions.tpp
#include "custom_types.h"
namespace loss {

   
    // MSE
    template <typename Scalar>
    Scalar MSELoss<Scalar>::compute(const MatrixD<Scalar>& reconstructed,
                                    const MatrixD<Scalar>& target) {
        return (reconstructed - target).squaredNorm() / static_cast<Scalar>(reconstructed.rows());
    }

    //SSR MSE
    template <typename Scalar>
    Scalar SSRMSELoss<Scalar>::compute( const MatrixD<Scalar>& reconstructed, const std::vector<SingleSparseRow<Scalar>>& targets) {
        const int feature_size = reconstructed.cols();
        Scalar total_error = 0;
        
        #pragma omp parallel for reduction(+:total_error)
        for(int i = 0; i < config::Training__batch_size; ++i){
            const auto& target_row = targets[i];
            const auto& reconstructed_row = output.row(i);

            //sum squared error over non-zero entires
            for (int k = 0; k < target_row.nnz; ++k){
                int col = target_row.indices[k];
                Scalar val = target_row.values[k];
                Scalar difference = reconstruced_row[col] - val;
                total_error += difference * difference;
        
            }
        }
        return total_error / config::Training__Batch_size;
    }



    // BCE
    template <typename Scalar>
    Scalar BCELoss<Scalar>::compute(const MatrixD<Scalar>& input,
                                    const MatrixD<Scalar>& target,
                                    Scalar epsilon) {
        auto clipped = input.array().min(1 - epsilon).max(epsilon);
        return -((target.array() * clipped.log()) + ((1 - target.array()) * (1 - clipped).log())).sum() / static_cast<Scalar>(input.rows());
    }

    // KL Divergence
    template <typename Scalar>
    Scalar KLLoss<Scalar>::compute(const VectorD<Scalar>& mu,
                                    const VectorD<Scalar>& logvar) {
        return static_cast<Scalar>(-0.5) * (1 + logvar.array() - mu.array().square() - logvar.array().exp()).sum();
    }

} // namespace loss
