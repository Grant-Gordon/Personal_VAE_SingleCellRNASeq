//Trainer.tpp
#pragma once
#include <pybind11/pybind11.h> 
#include "BatchCreator.h"
#include "custom_types.h"
#include "get_ChunkExprCSR_from_npz.tpp"
#include "config.h"

template <typename Scalar>
Trainer(Module<Scalar>& model,
        Optimizer<Scalar>& optimizer,
        const std::vector<std::string> count_files_list,
        const std::vector<std::string> metadata_files_list //NOTE: metadata is not currently being handled anywhere 
):
    model(model),    
    optimizer(optimizer),
    count_files_list(count_files_list)
    metadata_files_list(metadata_files_list)
{}
void Trainer<Scalar>::train(){
    for(int epoch = 0; epoch < config::Training__epochs; ++epoch){
        //TODO: shuffle chunks 
        for(std::string count_file : count_files_list){
            ChunkExprCSR<Scalar> chunk_csr = get_ChunkExprCSR_from_npz(count_file);

            train_on_chunk(ChunkExprCSR);
        }
    }

}

//chunk level training
template <typename Scalar>
void Trainer<Scalar>::train_on_chunk(const ChunkExprCSR& chunk_csr){

    BatchCreator bc = BatchCreator(chunk_csr);

    while(!bc.all_chunks_preloaded){
        this->train_on_batch(bc.get_next_batch());
    }
}
//Batch level training
template <typename Scalar>
void Trainer<Scalar>::train_on_batch(const BatchCreator<Scalar>::Batch& batch){

    auto recontructed = model.forward_input(batch);
    Scalar loss = loss::SSRMSELoss<Scalar>::compute(reconstructed, batch);

    //TODO: add logging

    model.backward();
    optimizer.step(model);
}


