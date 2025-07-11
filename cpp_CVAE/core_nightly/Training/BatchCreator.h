//BatchCreator.h
#pragma once
#include <vector>
#include <pybind11/numpy.h>
#include <algorithm> //std::shuffle
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "custom_types.h"

template <typename Scalar>
class BatchCreator{
    using Batch = typename std::vector<std::unique_ptr<SingleSparseRow<Scalar>>>;

    public:
        
        BatchCreator(
            const ChunkExprCSR<Scalar>& chunk_csr,
            const int num_batches_to_preload,
            const int batch_size,
            const int seed 
        );
        
        ~BatchCreator();
    
        //getters
        bool all_batches_preloaded() const {return all_batches_preloaded;}
        
    private:
        
        ChunkExprCSR<Scalar> chunk_csr;
        int batch_size;
        int total_batches_loaded;
        int num_batches_in_chunk;
        int num_batches_to_preload;
        
        std::vector<int> flat_chunk_sample_ids;     //contiquous buffer of shuffled chunk sample indices
        std::vector<int*> shuffled_split_batch_ids; // ptrs to the shuffled indices
        
        std::queue<Batch> preloaded_batch_queue;
        std::mutex queue_mutex;                     //prevents race conditions for pushing/popping from queue
        std::condition_variable queue_cv;           //notifies the training thread when queue has something in it to prevent inefficent sleep behavior 
        bool stop_flag;                             //Needed to gracefully destruct thread
        std::thread preload_thread; 
        bool all_batches_preloaded;

        int seed;//TODO need to figure out configs and where stuff like seed and batchsize is owned/defined
        int final_batch_size;

        void preload_batches();
        void generate_shuffled_split_batch_ids();
        const Batch& generate_batch(int* batch_sample_ids, int actual_batch_size); //Thread target, pushes to preloaded_batch queue, 
}

#include "BatchCreator.tpp"
