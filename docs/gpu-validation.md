# GPU acceptance procedure

Status: not executed. This is an acceptance gate, not a benchmark report.

1. Clone the public repository into a clean Python virtual environment on an NVIDIA GPU host. Install the correct CUDA-enabled PyTorch build from the official selector, then `pip install -e ".[video]"`.
2. Save `dehcode doctor --json`. Confirm the actual GPU, VRAM, available RAM and `torch.cuda_available`. `driver_cuda_max` is the driver's supported CUDA version, not the installed toolkit or the PyTorch build. Device IDs passed to `--gpu` refer to devices visible to PyTorch (respect CUDA_VISIBLE_DEVICES).
3. Run `dehcode recommend --json`; unknown requirements must remain unknown until measured for the specific profile.
4. Run `dehcode install wan-1.3b`. Record the returned revision. Run it a second time and confirm the same snapshot without another download.
5. First exercise plumbing with `dehcode run wan-1.3b --prompt "A cat walks through a garden" --width 256 --height 256 --frames 17 --steps 2 --output outputs/smoke.mp4`. This reduced sample is not a quality evaluation.
6. Verify the file decodes with ffprobe or a video player, frame count/dimensions/FPS match, frames are present, and the adjacent metadata contains the correct revision/seed. An exit code alone is insufficient.
7. Run defaults (832×480, 81 frames, 50 steps) with model-offload and enough RAM. Capture wall time and peak GPU/system memory using external monitoring. Watch for OOM; do not rerun larger settings blindly.
8. Repeat using cached weights with networking disabled. Exercise a portrait 432×768 configuration separately.
9. Record `python -m pip freeze`, DehCode commit, driver, hardware, profile, prompt/settings and measured results. Redact private paths and credentials before publishing a report.
10. Only mark a specific profile validated after a successful real run. Do not generalize measurements to different hardware, drivers, shapes, durations or precision.

The implementation session had Linux, approximately 15.64 GiB RAM, no detected NVIDIA GPU, and no installed PyTorch. Offline tests use fake lightweight dependencies; they are not generated videos.
