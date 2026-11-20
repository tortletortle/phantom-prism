import tempfile
import unittest
from pathlib import Path
from PIL import Image
from inspection_preflight import audit


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = []
        for i, split in enumerate(('train', 'validation', 'test')):
            name = f'{i}.png'
            Image.new('RGB', (16, 16), (i*60, 0, 0)).save(self.root/name)
            self.rows.append(dict(id=str(i), split=split, scene=f's{i}', image=name,
                                  objects=[dict(label='part', box=[0, 0, 8, 8])]))

    def test_valid(self):
        self.assertEqual(audit(self.root, self.rows)['test'], dict(images=1, objects=1, scenes=1))

    def test_scene_leakage(self):
        self.rows[2]['scene'] = self.rows[0]['scene']
        with self.assertRaisesRegex(ValueError, 'Scene leakage'):
            audit(self.root, self.rows)

    def test_pixel_duplicate(self):
        Image.open(self.root/'0.png').save(self.root/'copy.bmp')
        self.rows[2]['image'] = 'copy.bmp'
        with self.assertRaisesRegex(ValueError, 'Duplicate decoded'):
            audit(self.root, self.rows)

    def test_bad_box(self):
        self.rows[0]['objects'][0]['box'] = [0, 0, 17, 8]
        with self.assertRaisesRegex(ValueError, 'outside image'):
            audit(self.root, self.rows)

    def test_nonfinite_box(self):
        self.rows[0]['objects'][0]['box'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid box'):
            audit(self.root, self.rows)

    def test_missing(self):
        self.rows[0]['image'] = '../missing.png'
        with self.assertRaisesRegex(ValueError, 'Missing image'):
            audit(self.root, self.rows)

    def test_missing_positive(self):
        self.rows[2]['objects'] = []
        with self.assertRaisesRegex(ValueError, 'annotated positives'):
            audit(self.root, self.rows)

    def test_duplicate_id(self):
        self.rows[2]['id'] = self.rows[0]['id']
        with self.assertRaisesRegex(ValueError, 'Duplicate id'):
            audit(self.root, self.rows)

if __name__ == '__main__':
    unittest.main()
