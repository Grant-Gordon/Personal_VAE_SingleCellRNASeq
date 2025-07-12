import os
import subprocess
import argparse

import "config_parser"
import "generate_config_hpp"


def build_metadata_vocab(config):

def build_cpp_backend(build_dir="build"):
    os.makedirs(build_dir, exist_ok=True)
    subprocess.run(["cmake", ".."], cwd=build_dir, check=True)
    subprocess.run(["make", "-j"], cwd=build_dir, check=True)

def run_cpp_executable(build_dir="build", binar_name="main.cpp"):
    binary_path = os.path.join(build_dir, binary_name)
    subprocess.run([binary_path], check=True)
    


if __name__ == "main":
    config_path = "config.yaml"
    config = config_parser(config_path)
    
    build_metadata_vocab(config)
    generate_config_hpp(config)
    build_cpp_backend()
    run_cpp_executable()