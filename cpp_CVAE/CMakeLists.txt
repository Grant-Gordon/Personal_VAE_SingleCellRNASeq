cmake_minimum_required(VERSION 3.14)
project(CVAE_CPP)

set(CMAKE_CXX_STANDARD 17)

# Include header files
include_directories(
    ${PROJECT_SOURCE_DIR}/core_nightly/Layers
    ${PROJECT_SOURCE_DIR}/core_nightly/Layers/Activation_Layers
    ${PROJECT_SOURCE_DIR}/core_nightly/Loss
    ${PROJECT_SOURCE_DIR}/core_nightly/Module
    ${PROJECT_SOURCE_DIR}/core_nightly/Optimizer
    ${PROJECT_SOURCE_DIR}/core_nightly/PoC_trainer
    ${PROJECT_SOURCE_DIR}/external/eigen
)

# Add your executable
add_executable(PoC_trainer_exec
    core_nightly/PoC_trainer/main.cpp
)
