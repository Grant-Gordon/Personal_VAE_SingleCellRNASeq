//Trainer.tpp
#pragma once
#include <pybind11> 
#include "BatchCreator.h"
#include "custom_types.h"

template <typename Scalar>
Trainer(Module<Scalar>& model,
        Optimizer<Scalar>& optimizer,
        int batch_size,
        int num_features,
        int num_epochs
):
    model(model),    
    optimizer(optimizer),
    batch_size(batch_size),
    num_features(num_features)
    num_epochs(num_epochs)
{}
void train(){
    for(epoch : this->num_epochs){
        for(chunk file in data_dir){
            pythonChunk = frontend::load_chunk_npz();
            
            ChunkExprCSR chunk =(pythonchunl.col, pythonchunk.indptr, pythonchunk.vals, pythonchunk.size, pythonchunk.nnz);

            train_on_chunk(ChunkExprCSR);
        }
    }
}
//chunk level training
template <typename Scalar>
void train_on_chunk(const ChunkSparseCSR& chunk_csr){

    BatchCreator bc = BatchCreator(chunk_csr, config.num_batches_to_preload, this->batch_size, config.seed);

    while(!bc.all_chunks_preloaded){
        this->train_on_batch(bc.get_next_batch());
    }
}



void train_on_batch(const BatchCreator::Batch& batch){

    auto recontructed = model.forward_input(batch);
    Scalar loss = loss::SSRMSELoss<Scalar>::compute(reconstructed, batch);

    //TODO: add logging

    model.backward();
    optimizer.step(model);
}


