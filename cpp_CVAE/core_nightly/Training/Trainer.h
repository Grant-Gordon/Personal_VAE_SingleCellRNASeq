//Trainer.h

#pragma once

#include <vector>
#include "custom_types.h"
#include "loss_functions.h"
#include "Module.h"
#include "Optimizer.h"
#include "BatchCreator.h"
#include "config.hpp"
#include <Eigen/Sparse>

template <typename Scalar>
class Trainer{
    public:
        Trainer(Module<Scalar>& model,
            Optimizer<Scalar>& optimizer,
            int batch_size,
            // int num_features TODO: See if I can do some const expr stuff with this. 
        );
        ~Trainer() = default;
        void train();

    private:
        Module<Scalar>& model_;
        Optimizer<Scalar>& optimizer_;
        int batch_size_;
        int num_features_;
            
        void train_on_chunk(ChunkExprCSR chunk_csr);
        void train_batch(const std::vector<SingleSparseRow<Scakar>>& batch);
}


#include "Trainer.tpp"
