//BatchCreator.h
#pragma once
#include <vector>
#include <algorithm> //std::shuffle
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "custom_types.h"
#include "config.h"

template <typename Scalar>
class BatchCreator{
    public:    
        BatchCreator(const ChunkExprCSR<Scalar>& chunk_csr);
        
        ~BatchCreator();
    
        //getters
        bool all_batches_preloaded() const {return all_batches_preloaded;}
        
    private:
        
        ChunkExprCSR<Scalar> chunk_csr;
        int total_batches_loaded;
        int num_batches_in_chunk;
        
        std::vector<int> flat_chunk_sample_ids;     //contiquous buffer of shuffled chunk sample indices
        std::vector<int*> shuffled_split_batch_ids; // ptrs to the shuffled indices
        
        std::queue<Batch> preloaded_batch_queue;
        std::mutex queue_mutex;                     //prevents race conditions for pushing/popping from queue
        std::condition_variable queue_cv;           //notifies the training thread when queue has something in it to prevent inefficent sleep behavior 
        bool stop_flag;                             //Needed to gracefully destruct thread
        std::thread preload_thread; 
        bool all_batches_preloaded;

        //TODO: not sure this is still needed
        int final_batch_size;

        void preload_batches();
        void generate_shuffled_split_batch_ids();
        const Batch& generate_batch(int* batch_sample_ids, int actual_batch_size); //Thread target, pushes to preloaded_batch queue, 
}

#include "BatchCreator.tpp"
