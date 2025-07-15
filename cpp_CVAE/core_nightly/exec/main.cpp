#include <type_traits>
#include "config.h"
#include "Layer_all.h"
#include "Optimizer_all.h"
#include "Module.h"
#include "Trainer.h"
#include "utils_all.h"
#include "custom_types.h"

int main(){

    using scalar = std::remove_const_t<decltype(config::Global__scalar)>;

    std::vector<std::string> counts_files_list = get_matching_files(std::string(config::Data__data_dir), std::string(config::Data__counts_file_pattern)); //TODO: why doesn't this function just accept string_view. if were going for compile time, lets keep it compile time. 
    std::vector<std::string> metadata_files_list = get_matching_files(std::string(config::Data__data_dir), std::string(config::Data__metadata_file_pattern));

    Module<scalar> model = Module<scalar>(config::Model__basic_auto_encoder);
    Optimizer<scalar> optim = config::Training__optimizer;
    
    Trainer<scalar> trainer(model, optim, counts_files_list, metadata_files_list);
    trainer.train();

    return 0;

}