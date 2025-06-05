from utils.config_parser import parse_config
from training.trainer import Trainer

def main():
    config = parse_config()
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()