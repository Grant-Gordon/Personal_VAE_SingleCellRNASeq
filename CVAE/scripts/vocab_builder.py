import os
import pickle
import pandas as pd
import yaml
from collections import defaultdict
from utils.config_parser import parse_config


def build_vocab(config):
    data_dir = config["data"]["data_dir"]
    species = config["data"]["species"]
    save_dir = config["data"]["vocab_builder_out"]

    metadata_fields = config["metadata_fields"]
    ignored_fields = [field for field, meta in metadata_fields.items() if meta["type"] ==  "IGNORE"]
  
    os.makedirs(save_dir, exist_ok=True)
    vocab_path = os.path.join(save_dir, f"{species}_vocab_dict.pkl")

    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            existing_vocab = pickle.load(f)
        existing_fields = set(existing_vocab.keys())
        expected_fields = set(field for field, meta in metadata_fields.items() if meta["type"] in ["embedding", "onehot"])
        if existing_fields == expected_fields:
            print(f"Vocab for species '{species}' at '{vocab_path}' already exists with desired fields. Skipping Rebuild...")
            return vocab_path
        else:
            print(f"#############################################################\n"
                  "WARNING: Existing vocab file at '{vocab_path} does not match desired fields in config. REBUILDING...\n"
                  "#############################################################\n")
    
    field_values = defaultdict(set)

    for filename in os.listdir(data_dir):
        if not filename.startswith(f"{species}_metadata") or not filename.endswith(".pkl"):
            continue
    
        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
            assert isinstance(df, pd.DataFrame), f"Assert Error, expected type pandas.dataFrame from file at '{file_path}'"

            for field in df.columns:
                if field in ignored_fields:
                    continue
                for val in df[field].dropna().unique():
                    if isinstance(val, (str, int)):
                        field_values[field].add(val)
    
    
    vocab_dict = {}

    for field, values in field_values.items():
        sorted_values = sorted(list(values))
        vocab_dict[field] = {v : i for i,v in enumerate(sorted_values)}
        print(f"Built vocab for '{field}' with {len(sorted_values)} values")

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab_dict, f)
        print(f"Saved vocab to {vocab_path}")
    return vocab_path


if __name__ == "__main__":
    config = parse_config()
    build_vocab(config)
