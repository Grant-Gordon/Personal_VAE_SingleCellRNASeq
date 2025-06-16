from utils.config_parser import parse_config
from cf_discriminator.training.DiscriminatorTrainer import DiscriminatorTrainer
import torch

def main():
    config = parse_config()
    cvae_checkpoint_path = config["source_cvae"]["cvae_checkpoint_path"]
    assert cvae_checkpoint_path is not None, "Must specify 'cvae_checkpoint_path' in config."
    cvae = torch.load(cvae_checkpoint_path, map_location=config["training"]["device"])

    trainer = DiscriminatorTrainer(config=config, live_cvae=cvae)
    trainer.train_using_pretrained_cvae()


if __name__ == "__main__":
    main()