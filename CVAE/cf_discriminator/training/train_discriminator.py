from utils.config_parser import parse_config
from cf_discriminator.training.DiscriminatorTrainer import DiscriminatorTrainer

def main():
    config = parse_config()
    trainer = DiscriminatorTrainer(config)
    trainer.train_discriminator()

if __name__ == "__main__":
    main()