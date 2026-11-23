import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from pcb_reference import detect, run


class PCBTests(unittest.TestCase):
    def test_identity(self):
        a = np.zeros((32, 32), np.uint8)
        self.assertEqual(detect(a, a), [])

    def test_detect(self):
        a = np.zeros((32, 32), np.uint8)
        b = a.copy()
        b[8:16, 10:18] = 200
        b[1, 1] = 255
        result = detect(a, b)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], (10, 8, 18, 16))

    def test_bad_shape(self):
        with self.assertRaises(ValueError):
            detect(np.zeros((32, 32), np.uint8), np.zeros((16, 32), np.uint8))

    def test_bad_dtype(self):
        with self.assertRaises(ValueError):
            detect(np.zeros((32, 32)), np.zeros((32, 32)))

    def test_bad_threshold(self):
        with self.assertRaises(ValueError):
            detect(np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.uint8), 256)

    def test_bad_min_area(self):
        with self.assertRaises(ValueError):
            detect(np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.uint8), min_area=0)

    def test_file_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = np.zeros((32, 32), np.uint8)
            b = a.copy()
            b[8:16, 8:16] = 255
            Image.fromarray(a).save(root/'one_temp.jpg', quality=100)
            Image.fromarray(b).save(root/'one_test.jpg', quality=100)
            (root/'one.txt').write_text('8 8 16 16 1\n')
            split = root/'test.txt'
            split.write_text('one.jpg one.txt\n')
            result = run(root, split)
            self.assertEqual((result['images'], result['metrics_by_iou']['0.33']['tp'],
                              result['metrics_by_iou']['0.50']['tp']), (1, 1, 1))
            split.write_text('one.jpg one.txt\none.jpg one.txt\n')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                run(root, split)
            split.write_text('missing.jpg one.txt\n')
            with self.assertRaisesRegex(ValueError, 'Missing'):
                run(root, split)


if __name__ == '__main__':
    unittest.main()
