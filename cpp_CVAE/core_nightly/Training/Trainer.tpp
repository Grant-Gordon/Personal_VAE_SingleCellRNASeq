//Trainer.tpp
#pragma once
#include <pybind11/pybind11.h> 
#include "config_values.h"
#include "custom_types.h"
#include "BatchCreator.h"
#include "get_ChunkExprCSR_from_npz.tpp"
#include "macros.h"

template <typename Scalar>
Trainer<Scalar>::Trainer(
    std::unique_ptr<Module<Scalar>> model,
    std::unique_ptr<Optimizer<Scalar>> optimizer,
    std::vector<std::string> count_files_list,
    std::vector<std::string> metadata_files_list
    //NOTE: metadata is not currently being handled anywhere 
):
count_files_list(std::move(count_files_list)),
metadata_files_list(std::move(metadata_files_list))
{
    this->model = std::move(model);   
    this->optimizer = std::move(optimizer);
}

template <typename Scalar>
void Trainer<Scalar>::train(){
    VERBOSEL1("Inside Trainer::trian");
    for(int epoch = 0; epoch < configV::Training__epochs; ++epoch){
        VERBOSEL1("Starting Epoch: '"<<epoch << "'");
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
    VERBOSEL2("Inside Trainer::train_on_chunk");
    VERBOSEL2("Chunk Shape: ["<< chunk_csr.shape[0]<< ", " << chunk_csr.shape[1]<< "]");
    BatchCreator bc = BatchCreator(chunk_csr);
    bc.start_thread();

    while (true) {
        Batch<Scalar> batch = bc.get_next_batch();
        if (batch.empty()) break; //should only return empty once all batches have been trained on. See BatchCreator::get_next_batch()
        this->train_on_batch(batch);
    }
}   
//Batch level training
//TODO: realizing I unfortunatly kinda hardcoded this for SSRMSE
template <typename Scalar>
void Trainer<Scalar>::train_on_batch(const Batch<Scalar>& batch){
    VERBOSEL2("Inside Trainer::train_on_batch");
    auto reconstructed = model->forward(batch);
    Scalar loss = loss::SSRMSELoss<Scalar>::compute(reconstructed, batch);
    MatrixD<Scalar> loss_gradient = loss::SSRMSELoss<Scalar>::gradients(reconstructed, batch);
    VERBOSEL1("SSRMSELoss: " << loss);
    //TODO: add logging

    model->backward(loss_gradient);
    optimizer->step(model->get_layers_vector());
}


