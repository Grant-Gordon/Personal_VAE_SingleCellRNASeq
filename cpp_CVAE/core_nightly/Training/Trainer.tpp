//Trainer.tpp
#pragma once
#include <pybind11/pybind11.h> 
#include "config_values.h"
#include "custom_types.h"
#include "BatchCreator.h"
#include "get_ChunkExprCSR_from_npz.tpp"

template <typename Scalar>
Trainer<Scalar>::Trainer(Module<Scalar>& model,
        std::unique_ptr<Optimizer<Scalar>>& optimizer,
        const std::vector<std::string>& count_files_list,
        const std::vector<std::string>& metadata_files_list //NOTE: metadata is not currently being handled anywhere 
):
    model(model),    
    optimizer(optimizer),
    count_files_list(count_files_list),
    metadata_files_list(metadata_files_list)
{}

template <typename Scalar>
void Trainer<Scalar>::train(){
    for(int epoch = 0; epoch < configV::Training__epochs; ++epoch){
        //TODO: shuffle chunks 
        for(const std::string& count_file : count_files_list){
            ChunkExprCSR<Scalar> chunk_csr = get_ChunkExprCSR_from_npz<Scalar>(count_file);

            train_on_chunk(chunk_csr);
        }
    }

}

//chunk level training
template <typename Scalar>
void Trainer<Scalar>::train_on_chunk(const ChunkExprCSR<Scalar>& chunk_csr){

    BatchCreator bc = BatchCreator(chunk_csr);

    while(!bc.all_batches_preloaded){
        this->train_on_batch(bc.get_next_batch());
    }
}
//Batch level training
template <typename Scalar>
void Trainer<Scalar>::train_on_batch(const Batch<Scalar>& batch){

    auto reconstructed = model.forward(batch);
    Scalar loss = loss::SSRMSELoss<Scalar>::compute(reconstructed, batch);
    MatrixD<Scalar> loss_gradient = loss::SSRMSELoss<Scalar>::gradients(reconstructed, batch);

    //TODO: add logging

    model.backward(loss_gradient, batch); //TODO: loss is scalar but backwards takes vectorD grad output?
    optimizer->step(model.get_layers());
}


