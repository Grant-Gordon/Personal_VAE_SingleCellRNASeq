# compile_and_train.py
import os
import subprocess
import yaml
#from config.config_parser import parse_config
from config.generate_config_hpp import write_config_header


def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
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
    config_path = "./config/example_config.yaml"
    config = load_yaml(config_path)

    build_metadata_vocab(config)
    write_config_header(config_path, "config.h")  # external_context optional
    build_cpp_backend()
    #run_cpp_executable() #uncomment to train. 
