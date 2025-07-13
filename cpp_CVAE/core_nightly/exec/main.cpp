#pragma once

#include "config.h"
#include "Layer_all.h"
#include "Optimizer_all.h"
#include "Module.h"
#include "Trainer.h"
#include "utils_all.h"

void main(){

    std::vector<std::string> counts_files_list = generate_file_lists(config::data_dir, config::counts_glob);
    std::vector<std::string> metadata_files_list = generate_file_lists(config::data_dir, config::metadata_glob);

    Module<config::Global__scalar> model = Module<config::Gloabl__scalar>(config::Model_Architecture__layers_vector);
    Optimizer<config::Global__scalar> optim = config::Training__optimizer
    
    Trainer(model, optim, counts_files_list, metadata_files_list);
    Trainer.train();

}