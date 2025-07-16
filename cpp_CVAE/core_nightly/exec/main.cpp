#include <type_traits>
#include "config_objects.h"
#include "custom_types.h"
#include "Layer_all.h"
#include "Optimizer_all.h"
#include "Module.h"
#include "Trainer.h"
#include "utils_all.h"

int main(){

    using scalar = std::remove_const_t<decltype(configV::Global__scalar)>;
 
    const std::vector<std::string> counts_files_list = get_matching_files(std::string(configV::Data__data_dir), std::string(configV::Data__counts_file_pattern)); //TODO: why doesn't this function just accept string_view. if were going for compile time, lets keep it compile time. 
    const std::vector<std::string> metadata_files_list = get_matching_files(std::string(configV::Data__data_dir), std::string(configV::Data__metadata_file_pattern));

    Module<scalar> model(std::move(configO::get_Model__basic_auto_encoder<scalar>()));
    std::unique_ptr<Optimizer<scalar>> optim = configO::get_Training__optimizer<scalar>(); 
    
    Trainer<scalar> trainer(model, optim, counts_files_list, metadata_files_list);
    trainer.train();

    return 0;

}