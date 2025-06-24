cmake_minimum_required(VERSION 3.14)
project(CVAE_CPP)

set(CMAKE_CXX_STANDARD 17)

# Include header files
include_directories(
    ${PROJECT_SOURCE_DIR}/core_nightly/Layers
    ${PROJECT_SOURCE_DIR}/external/eigen

    )

# Add your executable
add_executable(cvae_test
    src/fafo_main.cpp
)
