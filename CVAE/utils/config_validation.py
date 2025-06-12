import os
import yaml
from typing import Dict, List, Tuple, Optional


def normalize_field_name(name:str) -> str:
    return name.strip().lower().replace("_", "")

def extract_metadata_fields(cvae_config: dict) -> Tuple[List[str], List[str]]:
    fields  = []
    ignored = []
    for field, spec in cvae_config.get("metadata_fields", {}).items(): #TODO spec why not config[metadata][type]
        norm = normalize_field_name(field)
        if isinstance(spec, dict):
            if spec.get("type", "").lower() == "ignore":
                ignored.append(norm)
            else:
                fields.append(norm)
        elif isinstance(spec, str):
            if spec.lower() == "ignore":
                ignored.append(norm)
            else:
                fields.append(norm)
    return fields, ignored

def validate_cvae_config(config:dict) -> None:
    required_keys = ["input_dim", "laten_dim", "metadata_fields", "batch_size"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in CVAE config:'{key}")
        
    if not isinstance(config["metadata_fields"], list):
        raise TypeError("'metadata_fields' should be a list of strings")


    