"""Pinned snapshots in the standard Hugging Face cache, no copied weights."""
import json
import os
from pathlib import Path
import shutil
import tempfile
from dehcode.config import home, cache


def manifest_path(model):
    return home()/'installed'/f"{model['id']}.json"


def installed(model):
    p = manifest_path(model)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if data['repository'] != model['repository']:
        return None
    root = Path(data['snapshot'])
    for entry in data['files']:
        f = root/entry['name']
        if not f.is_file() or (entry['size'] is not None and f.stat().st_size != entry['size']):
            return None
    return data


def install(model, revision=None):
    existing = installed(model)
    if existing and (revision is None or revision == existing['revision']):
        return existing
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as e:
        raise RuntimeError('Install video dependencies first: pip install -e ".[video]"') from e
    info = HfApi().model_info(model['repository'], revision=revision or 'main', files_metadata=True)
    # Only data files consumed by standard Diffusers classes; no remote Python.
    selected = [{'name':s.rfilename,'size':s.size} for s in info.siblings
                if Path(s.rfilename).suffix in {'.json','.safetensors','.txt','.model'}]
    if not any(e['name'].endswith('.safetensors') for e in selected):
        raise RuntimeError('No safetensors weights found in upstream snapshot.')
    target = cache()
    target.mkdir(parents=True, exist_ok=True)
    # Conservative full-size check. HF itself deduplicates shared blobs.
    sizes = [e['size'] for e in selected]
    if all(s is not None for s in sizes) and shutil.disk_usage(target).free < sum(sizes) + 2**30:
        raise RuntimeError('Insufficient free disk for a conservative full snapshot plus 1 GiB reserve.')
    path = snapshot_download(model['repository'], revision=info.sha, cache_dir=str(target),
                             allow_patterns=[e['name'] for e in selected])
    data = {'repository':model['repository'], 'revision':info.sha, 'snapshot':str(Path(path).resolve()), 'files':selected}
    for entry in selected:
        f = Path(path)/entry['name']
        if not f.is_file() or (entry['size'] is not None and f.stat().st_size != entry['size']):
            raise RuntimeError(f"Incomplete download: {entry['name']}")
    output = manifest_path(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output.parent, suffix='.json')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(data, stream, indent=2)
        os.replace(tmp, output)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return data
