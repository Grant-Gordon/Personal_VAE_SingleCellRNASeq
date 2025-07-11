//BatchCreator.tpp


using Batch = typename std::vector<std::unique_ptr<SingleSparseRow<Scalar>>>;

BatchCreator::BatchCreator(
    const ChunkExprCSR<Scalar>& chunk_csr,
):
    chunk_csr(chunk_csr),
    total_batches_loaded(0),
    all_batches_preloaded(false),
{
    this->num_batches_in_chunk = (this->chunk_csr.shape[0] + config::training_batch_size -1) / config::training_batch_size; //B=3 s=11, (11+2)/3 = 4
    this->final_batch_size = (this->chunk_csr.shape[0] % config::training_batch_size ==0) ? config::training_batch_size : this->chunk_csr.shape[0] % config::training_batch_size; // handles final batch 
    this->generate_shuffled_split_batch_ids();

    //NOTE prelaod_batches is only allcated 1 thread. 
    this->preload_thread = std::thread(&BatchCreator::preload_batches, this);
}

//TODO: create benchmark to determine if batches are consumed faster than created, if so add multiple preload threads rather than just one. 
void BatchCreator::preload_batches(){
    for(int i= 0; i < this->num_batches_in_chunk; ++i){
        if(this->stop_flag){break;}
        //TODO: add assert to confirm that it is only ever the final batch that this occures in. i.e. no wierd shuffling going on
        //Handles final batch where batch might be smaller than  batch_size
        int actual_batch_size = (i == this->num_batches_in_chunk -1) ? this->final_batch_size : config::training_batch_size;

        
        { //RAII, mutex is relased once scope ends
            std::unique_lock<std::mutex> lock(queue_mutex);
            //wait until there is room in the queue
            this->queue_cv.wait(lock, [this]() {
                return this->preloaded_batch_queue.size() < config::BatchCreator_num_batches_to_prelaod || this->stop_flag;
            });
            if (this->stop_flag){break;}
        }
        //dont need mutex for batch generation
        Batch batch = this->generate_batch(this->shuffled_split_batch_ids[i], actual_batch_size);
        {//do need mutex for pushing to queue 
            
            std::unique_lock<std::mutex> lock(queue_mutex);
            preloaded_batch_queue.push(std::move(batch));
            queue_cv.notify_all(); //Hey trainer, Theres a batch ready    
        }

        ++this->total_batches_loaded;
    }
    this->all_batches_preloaded = true;
}

//NOTE: actual_batch_size != batch_size, the final batch in a chunk may be smaller than batch_size if chunk_samples % batch_size != 0
Batch BatchCreator::generate_batch(int* batch_sample_ids, int actual_batch_size){
    // construct SSR samples corresponding to the batch sample ids in the chunk csr, and push to a vector
    Batch batch;
    batch.reserve(actual_batch_size); //TODO: size is not valid 

    //TODO: confirm this doesn't break with batches smaller than batchsize 
    //TODO: point of optimization - could thread this but need to be careful, might starve threads in forward pass. 
    for(int i = 0; i < actual_batch_size ; ++i){
        int row = batch_sample_ids[i];
        
        int start = chunk_csr.indptr_ptr_data[row];
        int end = chunk_csr.indptr_ptr_data[row + 1];
        int nnz = end - start;

        const int* indices = &chunk_csr.cols_ptr_data[start];
        const Scalar* data = &chunk_csr.vals_ptr_data[start];

        std::unique_ptr<SingleSparseRow<Scalar>> ssr = std::make_unique<SingleSparseRow<Scalar>>();
        ssr->indices = indices;
        ssr->data  = data;
        ssr-> nnz = nnz;
        
        batch.push_back(std::move(ssr));
    }
    
    return batch;
}

void BatchCreator::generate_shuffled_split_batch_ids(){
    int num_samples = this->chunk_csr->shape[0];

    this->flat_chunk_sample_id.resize(num_samples);

    std::iota(this->flat_chunk_sample_id.begin(), this->flat_chunk_sample_id.end(), 0);
    std::mt19927 rng(config::SEED);
    std::shuffle(this->flat_chunk_sample_id.begin(), this->flat_chunk_sample_id.end(), rng);

    this->shuffled_split_batch_ids.reserve(this->num_batches_in_chunk);
    
    //TODO: point of optimization - could thread this but need to be careful, might starve threads in forward pass. 
    for (int i = 0; i < this->num_batches_in_chunk; ++i){
        this->shuffled_split_batch_ids.push_back(&this->flat_chunk_sample_id[i * config::training_batch_size]);
    }
}


BatchCreator::Batch BatchCreator::get_next_batch(){
    std::unique_lock<std::mutex> lock(this->queue_mutex); //RAII
    queue_cv.wait(lock, [this](){ //[&] means capture all local vars by reference (local vars visible to lambda)
        return !preloaded_batch_queue.empty() || all_batches_preloaded || this->stop_flag; //wait until not empty or finished 
    });
    
    
    //TDOO: Ensure Trainer handles empty return when finished w chunk
    if (preloaded_batch_queue.empty()){
        return{}; //Done with chunk
    }

    Batch batch = std::move(this->preloaded_batch_queue.front());
    this->preloaded_batch_queue.pop();
    this->queue_cv.notify_all(); //Wakeup preloader just incase
    return batch;
}


BatchCreator::~BatchCreator(){
    this->stop_flag = true;
    this->queue_cv.notify_all(); // wake up any sleeping threads for clean exiting
    if(this->preload_thread.joinable()){
        this->preload_thread.join();
    }
}

