import os
import click
from .datasets import cleaner
from .utils import config


@click.command()
@click.option('--config-data', default=None)
@click.option('--clean-taxi', default='.')
def main(config_data, clean_taxi):
    if config_data:
        config.config_data(config_data)

    if clean_taxi:
        for f in os.listdir(clean_taxi):
            if f.endswith('.csv'):
                cleaner.clean_chitax_csv(os.path.join(clean_taxi, f))


if __name__ == '__main__':
    main()
