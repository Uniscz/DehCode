import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from dehcode.adapters.wan import WanAdapter
from dehcode.core.request import GenerationRequest
from dehcode.core.registry import get_model
from dehcode.core.download import install, installed
from dehcode.core.service import run


class PipelineTests(unittest.TestCase):
    def test_request_translation(self):
        r = GenerationRequest(prompt='A cat', negative_prompt='text', frames=17, steps=2, seed=7)
        adapter = WanAdapter()
        adapter.validate(r)
        args = adapter.arguments(r)
        self.assertEqual(args['num_frames'], 17)
        self.assertEqual(args['num_inference_steps'], 2)
        self.assertNotIn('fps', args)
        self.assertNotIn('seed', args)

    def test_reject_invalid_inputs(self):
        for kwargs in ({'prompt':''}, {'width':833}, {'frames':16}, {'frames':85},
                       {'fps':0}, {'steps':-1}, {'guidance':float('nan')}, {'seed':-1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                values = {'prompt':'A cat', **kwargs}
                WanAdapter().validate(GenerationRequest(**values))

    def test_portrait_is_accepted(self):
        WanAdapter().validate(GenerationRequest(prompt='A cat', width=480, height=832))

    def test_pin_cache_reuse_and_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'DEHCODE_HOME':tmp,'HF_HUB_CACHE':tmp+'/cache'}):
            root = Path(tmp)/'snapshot'
            root.mkdir()
            (root/'model_index.json').write_text('{}')
            (root/'weights.safetensors').write_bytes(b'test')
            info = types.SimpleNamespace(sha='a'*40, siblings=[
                types.SimpleNamespace(rfilename='model_index.json',size=2),
                types.SimpleNamespace(rfilename='weights.safetensors',size=4),
                types.SimpleNamespace(rfilename='untrusted.py',size=20)])
            api = types.SimpleNamespace(model_info=lambda *a, **k: info)
            fake = types.ModuleType('huggingface_hub')
            fake.HfApi = lambda: api
            calls = []
            def download(*a, **kwargs):
                calls.append(kwargs)
                return str(root)
            fake.snapshot_download = download
            with patch.dict('sys.modules', {'huggingface_hub':fake}):
                model = get_model('wan-1.3b')
                first = install(model)
                second = install(model)
                self.assertEqual(first, second)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]['revision'], 'a'*40)
                self.assertNotIn('untrusted.py', calls[0]['allow_patterns'])
                (root/'weights.safetensors').unlink()
                self.assertIsNone(installed(model))

    def test_missing_model_does_not_import_runtime(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'DEHCODE_HOME':tmp}):
            with self.assertRaisesRegex(RuntimeError, 'not installed'):
                run(get_model('wan-1.3b'), GenerationRequest(prompt='A cat'), 'x.mp4', 'model-offload', 0, print)

    def test_wan_loader_uses_local_safe_weights_and_float32_vae(self):
        from unittest.mock import MagicMock
        torch = types.ModuleType('torch')
        torch.float32 = 'float32'
        torch.bfloat16 = 'bf16'
        diffusers = types.ModuleType('diffusers')
        diffusers.AutoencoderKLWan = MagicMock()
        diffusers.WanPipeline = MagicMock()
        with patch.dict('sys.modules', {'torch':torch, 'diffusers':diffusers}):
            WanAdapter().load('/snapshot')
        kwargs = diffusers.AutoencoderKLWan.from_pretrained.call_args.kwargs
        self.assertTrue(kwargs['local_files_only'])
        self.assertTrue(kwargs['use_safetensors'])
        self.assertEqual(kwargs['torch_dtype'], 'float32')
        kwargs = diffusers.WanPipeline.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs['torch_dtype'], 'bf16')

    def test_runtime_refuses_cpu(self):
        from dehcode.runtimes.diffusers import execute
        torch = types.ModuleType('torch')
        torch.cuda = types.SimpleNamespace(is_available=lambda:False)
        utilities = types.ModuleType('diffusers.utils')
        utilities.export_to_video = lambda *a, **k: None
        with patch.dict('sys.modules', {'torch':torch, 'diffusers.utils':utilities}):
            with self.assertRaisesRegex(RuntimeError, 'CUDA-enabled'):
                execute(WanAdapter(), {}, GenerationRequest(prompt='A cat'), 'x.mp4','gpu',0,print)
