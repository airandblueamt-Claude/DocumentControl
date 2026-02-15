import os
import yaml


def load_config(config_path):
    """Load a YAML configuration file and return as dict.

    Falls back to the .example.yaml version if the real config doesn't exist
    (useful for Railway/Render deployments where secrets come from env vars).
    """
    if not os.path.exists(config_path):
        example_path = config_path.replace('.yaml', '.example.yaml')
        if os.path.exists(example_path):
            config_path = example_path
        else:
            raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Apply environment variable overrides for email config
    if 'server' in cfg or 'auth' in cfg:
        _apply_email_env_overrides(cfg)

    return cfg


def _apply_email_env_overrides(cfg):
    """Override email config values with environment variables if set.

    Supported env vars:
      EMAIL_IMAP_HOST, EMAIL_IMAP_PORT, EMAIL_USER, EMAIL_PASSWORD, EMAIL_ADDRESS
    """
    if os.environ.get('EMAIL_IMAP_HOST'):
        cfg.setdefault('server', {})['imap_host'] = os.environ['EMAIL_IMAP_HOST']
    if os.environ.get('EMAIL_IMAP_PORT'):
        cfg.setdefault('server', {})['imap_port'] = int(os.environ['EMAIL_IMAP_PORT'])
    if os.environ.get('EMAIL_USER'):
        cfg.setdefault('auth', {})['username'] = os.environ['EMAIL_USER']
        cfg.setdefault('auth', {})['auth_method'] = 'basic'
    if os.environ.get('EMAIL_PASSWORD'):
        cfg.setdefault('auth', {})['password'] = os.environ['EMAIL_PASSWORD']
    if os.environ.get('EMAIL_ADDRESS'):
        cfg.setdefault('mailbox', {})['email_address'] = os.environ['EMAIL_ADDRESS']


def resolve_path(path, base_dir):
    """Resolve a relative path against a base directory."""
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)
