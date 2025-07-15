#!/bin/bash
set -x
cd /home/grant/research/czi/Personal_VAE_SingleCellRNASeq
grep -R --include="config.h" "namespace config" .
find . -name config.h
grep -R --include="*.tpp" "#include \"config.h\"" .
g++ -E /home/grant/research/czi/Personal_VAE_SingleCellRNASeq/cpp_CVAE/core_nightly/Layers/LinearLayer.h -o preprocessed.cpp
g++ -E /home/grant/research/czi/Personal_VAE_SingleCellRNASeq/cpp_CVAE/core_nightly/exec/main.cpp-o preprocessed.cpp
cd /home/grant/research/czi/Personal_VAE_SingleCellRNASeq/cpp_CVAE/front_end_nightly
python3 -m exec.compile_and_train
head -n 20 make_output.log
cat /home/grant/research/czi/Personal_VAE_SingleCellRNASeq/cpp_CVAE/CMakeLists.txt