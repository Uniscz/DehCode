"""Shared Diffusers execution; no model-specific pipeline in the runtime."""
import json
import os
from pathlib import Path
import tempfile
from dataclasses import asdict


def execute(adapter, snapshot, request, output, profile, gpu, progress):
    adapter.validate(request)
    if profile not in ('gpu','model-offload','sequential-offload'):
        raise ValueError('Unsupported offload profile.')
    import torch
    from diffusers.utils import export_to_video
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA-enabled PyTorch and an NVIDIA GPU are required. Run dehcode doctor.')
    if not 0 <= gpu < torch.cuda.device_count():
        raise ValueError('GPU index is outside the devices visible to PyTorch.')
    output = Path(output).expanduser().resolve()
    if output.suffix.lower() != '.mp4':
        raise ValueError('Output must end in .mp4.')
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Output already exists. Choose a new filename.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.cuda.device(gpu):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('This adapter requires BF16 support on the selected GPU.')
        # Actual kernel dispatch catches incompatible CUDA architectures early.
        torch.ones(1, device=f'cuda:{gpu}', dtype=torch.bfloat16).add_(1)
        progress('Loading model from local snapshot')
        pipe = adapter.load(snapshot['snapshot'])
        tmp = None
        try:
            if profile == 'gpu':
                pipe.to(f'cuda:{gpu}')
            elif profile == 'model-offload':
                pipe.enable_model_cpu_offload(gpu_id=gpu)
            else:
                pipe.enable_sequential_cpu_offload(gpu_id=gpu)
            if hasattr(pipe.vae, 'enable_tiling'):
                pipe.vae.enable_tiling()
            def callback(_pipe, step, timestep, kwargs):
                progress(f'Denoising {step+1}/{request.steps}')
                return kwargs
            with torch.inference_mode():
                frames = pipe(**adapter.arguments(request),
                              generator=torch.Generator(device=f'cuda:{gpu}').manual_seed(request.seed),
                              callback_on_step_end=callback).frames[0]
            progress('Encoding MP4')
            fd, tmp = tempfile.mkstemp(dir=output.parent, suffix='.mp4')
            os.close(fd)
            export_to_video(frames, tmp, fps=request.fps)
            if Path(tmp).stat().st_size == 0:
                raise RuntimeError('Encoder produced an empty file.')
            # Exclusive creation prevents accidental overwrite by concurrent jobs.
            os.link(tmp, output)
            metadata = dict(request=asdict(request), repository=snapshot['repository'],
                            revision=snapshot['revision'], profile=profile, gpu=gpu,
                            torch=torch.__version__)
            output.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
            return str(output)
        finally:
            if tmp:
                Path(tmp).unlink(missing_ok=True)
            del pipe
            torch.cuda.empty_cache()
