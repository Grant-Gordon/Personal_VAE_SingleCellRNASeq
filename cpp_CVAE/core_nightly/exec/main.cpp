#include <type_traits>
#include "config_objects.h"
#include "custom_types.h"
#include "Layer_all.h"
#include "Optimizer_all.h"
#include "Module.h"
#include "Trainer.h"
#include "utils_all.h"
#include "macros.h"

int main(){
    VERBOSEL1("Executing main()");

    using scalar = std::remove_const_t<decltype(configV::Global__scalar)>;
 
    const std::vector<std::string> counts_files_list = get_matching_files(std::string(configV::Data__data_dir), std::string(configV::Data__counts_file_pattern)); //TODO: why doesn't this function just accept string_view. if were going for compile time, lets keep it compile time. 
    const std::vector<std::string> metadata_files_list = get_matching_files(std::string(configV::Data__data_dir), std::string(configV::Data__metadata_file_pattern));

    // auto model(std::move(configO::model));
    // auto optim(std::move(configO::optim)); 
    
    Trainer<scalar> trainer(std::move(configO::model), std::move(configO::optim), counts_files_list, metadata_files_list);
    VERBOSEL1("Starting Training");
    trainer.train();

    return 0;

}