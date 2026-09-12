import os
from pathlib import Path


def home():
    return Path(os.environ.get('DEHCODE_HOME', Path.home()/'.cache'/'dehcode')).expanduser().resolve()


def cache():
    # Honor an existing HF cache to avoid a second copy.
    return Path(os.environ.get('HF_HUB_CACHE', Path(os.environ.get('HF_HOME', home()/'huggingface'))/'hub')).expanduser()
