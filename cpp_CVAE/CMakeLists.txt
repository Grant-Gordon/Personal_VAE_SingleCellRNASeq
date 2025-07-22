cmake_minimum_required(VERSION 3.14)
project(CVAE_CPP)

set(CMAKE_CXX_STANDARD 17)
set(CONFIG_BUILD_DIR "" CACHE PATH "Path to generated config_[objects/values}_.h")

# Pybind / Python
set(PYBIND11_DIR ${CMAKE_SOURCE_DIR}/external/pybind11)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Add executable
add_executable(main_cpp
    core_nightly/exec/main.cpp
)

#Unit Test executable 
file(GLOB TEST_SOURCES
    core_nightly/exec/Tests/test_*.cpp 
    core_nightly/exec/Tests/run_all_tests.cpp 
)
add_executable(run_tests ${TEST_SOURCES})

# Target include directories
foreach(target main_cpp run_tests)
    target_include_directories(${target}
    BEFORE PRIVATE
        ${CONFIG_BUILD_DIR}
        ${PYBIND11_DIR}/include
        ${Python3_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/core_nightly/Layers
        ${PROJECT_SOURCE_DIR}/core_nightly/Layers/Linear_Layers
        ${PROJECT_SOURCE_DIR}/core_nightly/Layers/Activation_Layers
        ${PROJECT_SOURCE_DIR}/core_nightly/Loss
        ${PROJECT_SOURCE_DIR}/core_nightly/Module
        ${PROJECT_SOURCE_DIR}/core_nightly/Optimizer
        ${PROJECT_SOURCE_DIR}/core_nightly/Training
        ${PROJECT_SOURCE_DIR}/core_nightly/exec
        ${PROJECT_SOURCE_DIR}/core_nightly/utils
        ${PROJECT_SOURCE_DIR}/external/eigen
)
endforeach()



message(STATUS "Using CONFIG_BUILD_DIR = ${CONFIG_BUILD_DIR}")
