import unittest
import cv2
import numpy as np
from pcb_reference import detect
from prehardware_stress import make_reference
from pcb_registration import align_translation


class RegistrationTests(unittest.TestCase):
    def test_corrects_one_pixel_translation(self):
        ref = make_reference()
        shifted = cv2.warpAffine(ref, np.float32([[1,0,1],[0,1,0]]), (128,128), borderValue=255)
        self.assertGreater(len(detect(ref, shifted)), 0)
        aligned, meta = align_translation(ref, shifted)
        self.assertAlmostEqual(meta['dx'], 1.0, delta=.1)
        self.assertEqual(len(detect(ref, aligned)), 0)

    def test_preserves_six_by_six_defect(self):
        ref = make_reference()
        changed = ref.copy(); changed[40:46,70:76] = 0
        aligned, meta = align_translation(ref, changed)
        self.assertEqual(len(detect(ref, aligned)), 1)

    def test_rejects_shift_over_limit(self):
        ref = make_reference()
        shifted = cv2.warpAffine(ref, np.float32([[1,0,5],[0,1,0]]), (128,128), borderValue=255)
        with self.assertRaisesRegex(ValueError, 'max_shift'):
            align_translation(ref, shifted, max_shift=3)

    def test_rejects_low_correlation(self):
        ref = make_reference()
        unrelated = np.random.default_rng(99).integers(0,256,ref.shape,dtype=np.uint8)
        with self.assertRaises(ValueError):
            align_translation(ref, unrelated, min_correlation=.99)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            align_translation(np.zeros((10,10),np.uint8), np.zeros((9,10),np.uint8))


if __name__ == '__main__':
    unittest.main()
