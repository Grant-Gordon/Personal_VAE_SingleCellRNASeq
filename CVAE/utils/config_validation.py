import os
import yaml
from utils.config_parser import load_yaml_config
from typing import Dict, List, Tuple, Optional

#TODO finalize actual config
#TODO clean up/standardize usage of field normalization
#TODO validate file paths exist 

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
        "training" not in config or not isinstance(config["training"], dict)
        or "batch_size" not in config["training"]
        or "lr" not in config["training"]
        or "output_dir" not in config["training"]
        or "device" not in config["training"]
        or "lambda_l2_penalty" not in config["training"]
    ):
        raise ValueError("It is advised to include the following yaml config 'training:\n\tepochs:\n\tbatch_size\n\tdevice:\n\tlambda_l2_penalty:"
        "One or more of these fields were configured incorrectly"
        "\n This code has defaults in place for some of these values though it is not guaranteed for all scenarios")

    if "metadata_fields" not in config or not isinstance(config["metadata_fields"], dict):
        raise ValueError("CVAE config must include a 'metadata_fields' dictionary.")
    
    if (   
        "data" not in config or not isinstance(config["data"], dict)
        or "data_dir" not in config["data"]
        or "species" not in config["data"]
        or "chunks_preloaded" not in config["data"]
        or "num_preloader_threads" not in config["data"]
        or "vocab_builder_out" not in config["data"]
    ):
        raise ValueError("It is advised to inclued the following yaml config "
        "'data:\n\tdata_dir:\n\tspecies:\n\tchunks_preloaded:\n\tnum_preloader_threads:\n\tvocab_builder_out:'"
        "\n This code has defaults in place for some of these values though it is not gauranteed for all scenarios")
    
    #TODO confirm that config parser returns (key, None) for blank config fields
def validate_discriminator_config(discr_config: dict, cvae_config: dict) -> None:
    
    #Syntax check/ Confirm all fields exist
    if (
        "source_cvae" not in discr_config or not isinstance(discr_config["source_cvae"], dict)
        or"train_with_pretrained_cvae" not in discr_config["source_cvae"]
        or "cvae_checkpoint_path" not in discr_config["source_cvae"]
        or "cvae_config_path" not in discr_config["source_cvae"]
    ):
        raise ValueError("It is advised to inclued the following yaml config "
        "'source_cvae:\n\train_with_pretrained_cvae:\n\tcvae_checkpoint_path:\n\tcvae_config_path:'"
        "\n This code has defaults in place for some of these values though it is not guaranteed for all scenarios")

    if(
        "discriminator_architecture" not in discr_config or not isinstance(discr_config["discriminator_architecture"], dict)
        or "hidden_dim" not in discr_config["discriminator_architecture"]
        or "dropout" not in discr_config["discriminator_architecture"]
        or "activation" not in discr_config["discriminator_architecture"]
        or "learning_rate"  not in discr_config["discriminator_architecture"]
    ):
         raise ValueError("It is advised to inclued the following yaml config "
        "'discriminator_architecture:\n\thidden_dim:\n\tdropout:\n\tactivation:\n\tlearning_rate:\n'"
        "This code has defaults in place for some of these values though it is not guaranteed for all scenarios")
    if(
        "fields_to_change" not in discr_config or not isinstance(discr_config["fields_to_change"], dict)
    ):
        raise ValueError("config 'fields_to_change(dict)' is required but was not included. \nFor strictly trans generation leave this field blank")
    

    #Semantic checks
    #No pretrained but paths
    if(
        discr_config["source_cvae"]["train_with_pretrained_cvae"] == False and 
        (discr_config["source_cvae"]["cvae_checkpoint_path"] is not None or discr_config["source_cvae"]["cvae_config_path"] is not None)
    ):
        raise ValueError("Spec 'train_with_pretrained_cvae' was False, but values were provided for pretrained CVAE paths.\n"
            "These are mutually exclusive specs. 'cvae_checkpoint_path' and 'cvae_config_path' should be left empty")

    #Pretrained but No paths    
    if(
        discr_config["source_cvae"]["train_with_pretrained_cvae"] == True and 
        (discr_config["source_cvae"]["cvae_checkpoint_path"] is None or discr_config["source_cvae"]["cvae_config_path"] is None)
    ):
        raise ValueError("Spec 'train_with_pretrained_cvae' was True but no paths were specified for the pretrained CVAE\n")
    
    #Warn about mismatching fields:
    for field, spec  in cvae_config["metadata_fields"].items():
        if isinstance(spec,dict) and spec.get("type", "").lower() != "ignore":  #compare changed fields against cvae meta fields
            if field not in discr_config["fields_to_change"]: 
                print(f"WARNING: Field '{field}' is not included in discriminator config but used in CVAE training.\n Trans samples for field '{field} will not be generated\n")
        elif normalize_field_name(field) in discr_config["fields_to_change"]:
            raise ValueError(f"CVAE config has field '{field}' set to 'IGNORE' but the discriminator config contains field '{field}' in key 'fields_to_change\n'")
        
