import argparse
import json
import subprocess
import sys
import uuid
from dataclasses import fields
from importlib import metadata
from dehcode.hardware import detect
from dehcode.core.registry import models, get_model
from dehcode.core.recommend import recommend
from dehcode.core.request import GenerationRequest
from dehcode.config import home


def prepare_dependencies():
    # Resolve the declared video extra from this installed distribution, not PyPI dehcode.
    try:
        from packaging.requirements import Requirement
        declared = metadata.requires('dehcode') or []
    except (ImportError, metadata.PackageNotFoundError) as e:
        raise RuntimeError('Install the project first: python -m pip install -e .') from e
    dependencies = []
    for item in declared:
        requirement = Requirement(item)
        if requirement.marker and requirement.marker.evaluate({'extra':'video'}):
            requirement.marker = None
            dependencies.append(str(requirement))
    if not dependencies:
        raise RuntimeError('No video dependencies found in installed package metadata.')
    subprocess.run([sys.executable, '-m', 'pip', 'install', *dependencies], check=True)


def main():
    parser = argparse.ArgumentParser(prog='dehcode', description='FREE-FIRST local AI video runtime')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('doctor','models','recommend'):
        sub.add_parser(name).add_argument('--json', action='store_true')
    install = sub.add_parser('install', help='Download and pin a model snapshot')
    install.add_argument('model')
    install.add_argument('--revision', help='Optional upstream revision; resolved to immutable SHA')
    install.add_argument('--dependencies', action='store_true', help='Install video dependencies into the current Python environment')
    run = sub.add_parser('run', help='Generate an MP4 using installed local weights')
    run.add_argument('model')
    run.add_argument('--prompt', required=True)
    run.add_argument('--negative-prompt', default='')
    for name in ('width','height','frames','fps','steps'):
        run.add_argument('--'+name, type=int)
    run.add_argument('--seed', type=int, default=42)
    run.add_argument('--guidance', type=float)
    run.add_argument('--profile', choices=['gpu','model-offload','sequential-offload'], default='model-offload')
    run.add_argument('--gpu', type=int, default=0)
    run.add_argument('--output', help='New .mp4 output path')
    args = parser.parse_args()
    try:
        if args.command == 'doctor':
            value = detect()
        elif args.command == 'models':
            value = models()
        elif args.command == 'recommend':
            hardware = detect()
            value = [recommend(hardware, m) for m in models()]
        elif args.command == 'install':
            from dehcode.core.download import install as download
            model = get_model(args.model)
            if args.dependencies:
                prepare_dependencies()
            value = download(model, args.revision)
        else:
            from dehcode.core.service import run as generate
            model = get_model(args.model)
            params = {f.name: getattr(args, f.name) for f in fields(GenerationRequest)}
            for name, default in model['defaults'].items():
                if params[name] is None:
                    params[name] = default
            request = GenerationRequest(**params)
            output = args.output or home()/'outputs'/f'{uuid.uuid4().hex}.mp4'
            value = {'output':generate(model, request, output, args.profile, args.gpu,
                                      lambda msg: print(msg, file=sys.stderr, flush=True))}
        print(json.dumps(value, indent=2, ensure_ascii=False))
    except (ValueError, RuntimeError, OSError, ImportError, subprocess.CalledProcessError) as e:
        print(f'dehcode: {e}', file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print('dehcode: interrupted; completed downloads remain cached.', file=sys.stderr)
        raise SystemExit(130)


if __name__ == '__main__':
    main()
