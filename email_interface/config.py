import os
import yaml


def load_config(config_path):
    """Load a YAML configuration file and return as dict."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def resolve_path(path, base_dir):
    """Resolve a relative path against a base directory."""
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)
