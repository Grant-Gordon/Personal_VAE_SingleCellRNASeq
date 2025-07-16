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

def build_cpp_backend(build_dir, CMakeList_home_dir):
    os.makedirs(build_dir, exist_ok=True)
    subprocess.run(["cmake", f"-DCONFIG_BUILD_DIR={build_dir}", f"{CMakeList_home_dir}"], cwd=build_dir, check=True)
    subprocess.run(["make", "-j"], cwd=build_dir, check=True)


# #DEBUG BUILD
# def build_cpp_backend (build_dir: str, CMakeList_home_dir: str):
#     os.makedirs(build_dir, exist_ok=True)
#     with open("cmake_output.log", "w") as cmake_log:
#         subprocess.run(["cmake", f"-DCONFIG_BUILD_DIR={build_dir}", f"{CMakeList_home_dir}"], cwd=build_dir, stdout=cmake_log, stderr=subprocess.STDOUT, check=True)
#     with open("make_output.log", "w") as make_log:
#         subprocess.run(["make", "-j"], cwd=build_dir,stdout=make_log, stderr=subprocess.STDOUT, check=True)

def run_cpp_executable(build_dir, binary_name="main"):
    binary_path = os.path.join(build_dir, binary_name)
    subprocess.run([binary_path], check=True)

if __name__ == "__main__":
    config_yaml_path = "./config/example_config.yaml" #TODO: ARG parse this to prevent hardcoding config paths. 
    config = load_yaml(config_yaml_path)
    build_dir = config["global"]["build_dir"]
    CMakeLists_home_dir  = config["global"]["CMakeLists_home_dir"]


    os.makedirs(build_dir, exist_ok=True)

    build_metadata_vocab(config)
    write_config_header(config_yaml_path, f'{build_dir}/config_values.h', f'{build_dir}/config_objects.h')  # external_context optional
    build_cpp_backend(build_dir, CMakeLists_home_dir) 
    #run_cpp_executable(config["global"]["build_dir"], config["cpp_executable"]) #uncomment to train.  //TDOO: include cpp_executable name in config (i.e. update CMakeList with set() or smake -D{})
