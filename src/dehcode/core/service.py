from dehcode.adapters import get_adapter
from dehcode.core.download import installed


def run(model, request, output, profile, gpu, progress):
    adapter = get_adapter(model['adapter'])
    adapter.validate(request)
    snapshot = installed(model)
    if snapshot is None:
        raise RuntimeError(f"Model not installed or cache incomplete. Run dehcode install {model['id']}.")
    if model['backend'] != 'diffusers':
        raise ValueError(f"Runtime not implemented: {model['backend']}")
    from dehcode.runtimes.diffusers import execute
    return execute(adapter, snapshot, request, output, profile, gpu, progress)
