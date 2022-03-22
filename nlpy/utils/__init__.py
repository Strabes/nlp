import json

def load_config(config):
    with open(config,"r") as f:
        config = json.load(f)
    return config