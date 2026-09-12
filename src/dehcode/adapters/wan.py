class WanAdapter:
    def validate(self, request):
        request.validate()
        if request.width % 16 or request.height % 16:
            raise ValueError('Wan width and height must be multiples of 16.')
        if (request.frames - 1) % 4:
            raise ValueError('Wan frames must follow 4n+1, e.g. 17, 49 or 81.')
        if request.width * request.height > 832 * 480 or request.frames > 81:
            raise ValueError('This first adapter limits generation to 399360 pixels and 81 frames; higher settings are unvalidated.')

    def load(self, path):
        import torch
        from diffusers import AutoencoderKLWan, WanPipeline
        vae = AutoencoderKLWan.from_pretrained(path, subfolder='vae', torch_dtype=torch.float32, local_files_only=True, use_safetensors=True)
        return WanPipeline.from_pretrained(path, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True, use_safetensors=True)

    def arguments(self, request):
        return dict(prompt=request.prompt, negative_prompt=request.negative_prompt,
                    width=request.width, height=request.height, num_frames=request.frames,
                    num_inference_steps=request.steps, guidance_scale=request.guidance)
