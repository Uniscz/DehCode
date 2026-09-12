# Architecture
CLI → core orchestration → adapter → runtime. Hardware detection and the declarative model registry feed the recommendation engine. Configuration owns cache/output paths. Model adapters translate validated requests, while runtimes manage execution. Heavy ML imports must remain optional so doctor works without CUDA.

Planned runtimes: Diffusers first, then native Python and ComfyUI. No UI or ComfyUI implementation is claimed yet.
