import os
import click
import yaml


CONFIG = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'config.yaml')


def config_data(data=None):
    """ By checking config.yaml, return or overwrite the data folder location

    Args:
        data (str, optional): Defaults to None. The path to data folder

    Raises:
        Exception: If there is currently no data folder set in config

    Returns:
        str: The path to pre-defined data folder
    """
    with open(CONFIG, 'r') as f:
        settings = yaml.load(f)

    if data is None:
        if settings['data'] == '':
            raise Exception("You haven't set up the data folder yet")
    else:
        settings['data'] = data
        with open(CONFIG, 'w') as f:
            f.write(yaml.dump(settings, default_flow_style=False))

    return settings['data']


@click.command()
@click.option('-d', '--data', default=None)
def main(data):
    if data:
        config_data(data)


if __name__ == '__main__':
    main()
