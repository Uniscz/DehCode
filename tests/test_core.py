import unittest
from dehcode.hardware import parse_nvidia, detect
from dehcode.core.registry import models, get_model
from dehcode.core.recommend import recommend


class CoreTests(unittest.TestCase):
    def test_multiple_gpus_and_bad_rows(self):
        found = parse_nvidia('0, RTX 4090, 24564, 23000, 550.1\n1, RTX 3060, 12288, 10000, 550.1\ninvalid\n2, Broken, N/A, 0, x')
        self.assertEqual(len(found), 2)
        self.assertEqual(found[1]['vram_gib'], 12)

    def test_no_gpu(self):
        r = recommend({'gpus': [], 'torch': {}}, models()[0])
        self.assertEqual(r['status'], 'not-recommended')

    def test_unknown_is_not_compatible(self):
        hw = {'gpus':[{'index':0,'free_vram_gib':24}], 'torch':{'cuda_available':True}}
        r = recommend(hw, models()[0])
        self.assertEqual(r['status'], 'unknown')
        self.assertFalse(r['validated'])

    def test_registry_unknown_id(self):
        with self.assertRaises(ValueError):
            get_model('../../other')

    def test_doctor_shape(self):
        self.assertIn('driver_cuda_max', detect())
