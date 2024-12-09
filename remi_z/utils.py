import yaml

def read_yaml(fp):
    with open(fp, 'r') as f:
        data = yaml.safe_load(f)
    return data