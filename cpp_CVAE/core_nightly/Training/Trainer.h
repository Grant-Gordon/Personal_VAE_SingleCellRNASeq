//Trainer.h

#pragma once

#include <vector>
#include <Eigen/Sparse>
#include "loss_functions.h"
#include "Module.h"
#include "Optimizer_all.h"
#include "BatchCreator.h"
#include "custom_types.h"
#include "config.h"

template <typename Scalar>
class Trainer{
    public:
        Trainer(Module<Scalar>& model,
            Optimizer<Scalar>& optimizer,
            const std::vector<std::string>& count_files_list,
            const std::vector<std::string>& metadata_files_list
            // int num_features TODO: See if I can do some const expr stuff with this. 
        );
        ~Trainer() = default;
        void train();
        
        private:
        Module<Scalar>& model;
        Optimizer<Scalar>& optimizer;
        const std::vector<std::string>& count_files_list;
        const std::vector<std::string>& metadata_files_list;
        

        void train();
        void train_on_chunk(ChunkExprCSR chunk_csr);
        void train_batch(const std::vector<SingleSparseRow<Scakar>>& batch);
}


#include "Trainer.tpp"
