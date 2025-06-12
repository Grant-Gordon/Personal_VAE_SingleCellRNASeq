import os
import yaml
from utils.config_parser import load_yaml_config
from typing import Dict, List, Tuple, Optional


def normalize_field_name(name:str) -> str:
    return name.strip().lower().replace("_", "")

def extract_cvae_metadata_fields(cvae_config: dict) -> Tuple[List[str], List[str]]:
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


def validate_cvae_config(config: dict) -> None:
    if (
        "training" not in config
        or "batch_size" not in config["training"]
        or "lr" not in config["training"]
        or "output_dir" not in config["training"]
        or "device" not in config["training"]
        or "lambda_l2_penalty" not in config["training"]
    ):
        raise ValueError("It is advised to include the following yaml config 'training:\n\tepochs:\n\tbatch_size\n\tdevice:\n\tlambda_l2_penalty:"
        "One or more of these fields were configured incorrectly"
        "\n This code has defaults in place for some of these values though it is not guranteed for all scinarios")

    if "metadata_fields" not in config or not isinstance(config["metadata_fields"], dict):
        raise ValueError("CVAE config must include a 'metadata_fields' dictionary.")

    if "metadata_vobab" not in config or not isinstance(config["metadata_vocab"], str):
        raise ValueError("CVAE config requries")
