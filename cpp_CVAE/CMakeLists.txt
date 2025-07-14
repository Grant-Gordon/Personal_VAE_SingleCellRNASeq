cmake_minimum_required(VERSION 3.14)
project(CVAE_CPP)

set(CMAKE_CXX_STANDARD 17)


#pybind stuff
set(PYBIND11_DIR ${CMAKE_SOURCE_DIR}/external/pybind11)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Include header files
include_directories(
    ${PROJECT_BINARY_DIR}
    ${PYBIND11_DIR}/include
    ${Python3_INCLUDE_DIRS}

    ${PROJECT_SOURCE_DIR}/core_nightly/Layers
    ${PROJECT_SOURCE_DIR}/core_nightly/Layers/Activation_Layers
    ${PROJECT_SOURCE_DIR}/core_nightly/Loss
    ${PROJECT_SOURCE_DIR}/core_nightly/Module
    ${PROJECT_SOURCE_DIR}/core_nightly/Optimizer
    #${PROJECT_SOURCE_DIR}/core_nightly/PoC_trainer
    ${PROJECT_SOURCE_DIR}/core_nightly/Training
    ${PROJECT_SOURCE_DIR}/core_nightly/exec
    ${PROJECT_SOURCE_DIR}/core_nightly/utils
    ${PROJECT_SOURCE_DIR}/external/eigen
)

# Add your executable
add_executable(PoC_trainer_exec
    core_nightly/exec/main.cpp
)
