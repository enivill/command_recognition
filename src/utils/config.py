import yaml
import os

config_path = 'src/configs/'
config = {}


# Function to load yaml configuration file
def load_config(config_name: str):
    with open(os.path.join(config_path, config_name)) as file:
        print(file)
        global config
        config = yaml.safe_load(file)


def get_config():
    return config

