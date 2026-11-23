import unittest
from prehardware_stress import run


class PreHardwareCharacterization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = run()['cases']

    def test_identical_pair_has_no_detection(self):
        self.assertEqual(self.results['identical']['detections'], 0)

    def test_four_by_four_defect_is_filtered(self):
        self.assertEqual(self.results['defect_4x4']['detections'], 0)

    def test_six_by_six_defect_is_detected(self):
        self.assertEqual(self.results['defect_6x6']['detections'], 1)

    def test_one_pixel_registration_error_causes_false_regions(self):
        self.assertGreater(self.results['translation_1px']['detections'], 0)

    def test_large_illumination_shift_causes_false_regions(self):
        self.assertGreater(self.results['brightness_plus40']['detections'], 0)

    def test_blur_causes_false_regions(self):
        self.assertGreater(self.results['blur_3x3_sigma0.8']['detections'], 0)

    def test_moderate_brightness_shift_is_below_threshold(self):
        self.assertEqual(self.results['brightness_plus25']['detections'], 0)

    def test_noise_sigma20_fixture(self):
        self.assertEqual(self.results['noise_sigma20']['detections'], 0)


if __name__ == '__main__':
    unittest.main()
