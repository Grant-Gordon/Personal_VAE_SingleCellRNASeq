import yaml
import argparse


def load_yaml_config(path):
    with open (path, 'r') as f:
        return yaml.safe_load(f)
    
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cvae_train.yaml", help="Path to config file")
    args, overrides = parser.parse_known_args()
    config = load_yaml_config(args.config)
    if not isinstance(config, dict):
        raise TypeError(f"Expected config to be dict, got {type(config)} instead.")

    
    # Allow dot-notation CLI overrides: training.epochs=5 TODO???
    for override in overrides:
        key, val = override.split('=')
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = yaml.safe_load(val)  # convert string to int/float/bool
    return config