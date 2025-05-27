from dataloader.loader import create_dataloader
import time
from threading import Thread

def start_preload_workers(dataset, data_dir, batch_size, num_threads, preload_buffer, buffer_lock, chunk_queue, config):
    def worker():
        while True:
            chunk_idx = chunk_queue.get()
            counts_file, metadata_file = dataset[chunk_idx]
            start_time = time.time()
            loader = create_dataloader(data_dir, counts_file, metadata_file, batch_size, config)
            with buffer_lock:
                preload_buffer[chunk_idx] = (loader, time.time() - start_time)
            chunk_queue.task_done()
        
    for _ in range(num_threads):
        t = Thread(target=worker)
        t.daemon=True
        t.start()