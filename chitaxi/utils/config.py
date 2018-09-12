import os
import yaml


CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    'config.yaml')


def get_config():
    with open(CONFIG, 'r') as f:
        settings = yaml.load(f)

    return settings


def get_path_taxi():
    settings = get_config()
    return os.path.join(settings['data'],
                        settings['file_names']['taxi'])


def config_data(data=None):
    """ By checking config.yaml, return or overwrite the data folder location

    Args:
        data (str, optional): Defaults to None. The path to data folder

    Raises:
        Exception: If there is currently no data folder set in config

    Returns:
        str: The path to pre-defined data folder
    """
    settings = get_config()

    if data is None:
        if settings['data'] == '':
            raise Exception("You haven't set up the data folder yet")
    else:
        settings['data'] = data
        with open(CONFIG, 'w') as f:
            f.write(yaml.dump(settings, default_flow_style=False))

    return settings['data']
