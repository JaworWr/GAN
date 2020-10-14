import argparse
from typing import Type

from base.data_loader import BaseDataLoaderFactory
from base.discriminator import BaseDiscriminator
from base.generator import BaseGenerator
from base.trainer import BaseTrainer
from utils.factory import get_class
from utils.utils import read_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the configuration", required=True, type=str)
    parser.add_argument("--disc-save-path", help="Discriminator save path", type=str)
    parser.add_argument("--disc-load-path", help="Discriminator load path", type=str)
    parser.add_argument("--gen-save-path", help="Generator save path", type=str)
    parser.add_argument("--gen-load-path", help="Generator load path", type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    config = read_config(args.config)

    disc_cls: Type[BaseDiscriminator] = get_class(config.discriminator.module_name, config.discriminator.class_name)
    gen_cls: Type[BaseGenerator] = get_class(config.generator.module_name, config.generator.class_name)
    discriminator = disc_cls(config)
    if args.disc_load_path is not None:
        discriminator.load(args.disc_load_path)
    generator = gen_cls(config)
    if args.gen_load_path is not None:
        generator.load(args.gen_load_path)

    dl_cls: Type[BaseDataLoaderFactory] = get_class(config.data.module_name, config.data.class_name)
    dl = dl_cls.get_data_loader(config)

    trainer_cls: Type[BaseTrainer] = get_class(config.trainer.module_name, config.trainer.class_name)
    trainer = trainer_cls(config, dl, discriminator, generator, disc_save_path=args.disc_save_path,
                          disc_load_path=args.disc_load_path)

    print("Training...")
    try:
        trainer.train()
    finally:
        print("Saving...")
        if args.disc_save_path is not None:
            discriminator.save(args.disc_save_path)
        if args.gen_save_path is not None:
            generator.save(args.gen_save_path)


if __name__ == '__main__':
    main()
