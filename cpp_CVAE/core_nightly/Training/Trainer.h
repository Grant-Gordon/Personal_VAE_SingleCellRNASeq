//Trainer.h

#pragma once

#include <vector>
#include <Eigen/Sparse>
#include "config.h"
#include "custom_types.h"
#include "loss_functions.h"
#include "Module.h"
#include "Optimizer_all.h"
#include "BatchCreator.h"

template <typename Scalar>
class Trainer{
    public:
        Trainer(Module<Scalar>& model,
            Optimizer<Scalar>& optimizer,
            const std::vector<std::string>& count_files_list,
            const std::vector<std::string>& metadata_files_list
            // int num_features TODO: See if I can do some const expr stuff with this. 
        );
        
        void train();
        
        ~Trainer() = default;
        
        private:
            Module<Scalar>& model;
            Optimizer<Scalar>& optimizer;
            const std::vector<std::string>& count_files_list;
            const std::vector<std::string>& metadata_files_list;
            

            void train_on_chunk(const ChunkExprCSR<Scalar>& chunk_csr);
            void train_on_batch(const Batch<Scalar>& batch);
};


#include "Trainer.tpp"
