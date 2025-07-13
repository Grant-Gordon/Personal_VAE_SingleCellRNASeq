# compile_and_train.py
import os
import subprocess
from config import config_parser
from config.generate_config_hpp import write_config_header

def build_metadata_vocab(config):
    # TODO: implement metadata vocabulary builder
    pass

def build_cpp_backend(build_dir="build"):
    os.makedirs(build_dir, exist_ok=True)
    subprocess.run(["cmake", ".."], cwd=build_dir, check=True)
    subprocess.run(["make", "-j"], cwd=build_dir, check=True)

def run_cpp_executable(build_dir="build", binary_name="main"):
    binary_path = os.path.join(build_dir, binary_name)
    subprocess.run([binary_path], check=True)

if __name__ == "__main__":
    config_path = "config.yaml"
    config = config_parser(config_path)

    build_metadata_vocab(config)
    write_config_header(config_path, "config.h")  # external_context optional
    build_cpp_backend()
    run_cpp_executable()
