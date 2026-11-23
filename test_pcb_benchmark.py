import tempfile
import unittest
from pathlib import Path
from PIL import Image
from pcb_benchmark import benchmark, git_blob, snapshot


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        a = Image.new('L', (32,32))
        b = a.copy()
        b.paste(255, (8,8,16,16))
        a.save(self.root/'one_temp.jpg', quality=100)
        b.save(self.root/'one_test.jpg', quality=100)
        (self.root/'one.txt').write_text('8 8 16 16 1\n')
        self.split = self.root/'test.txt'
        self.raw = b'one.jpg one.txt\n'
        self.split.write_bytes(self.raw)

    def test_fixture_report(self):
        r = benchmark(self.root, self.split, git_blob(self.raw), 1)
        self.assertEqual(r['metrics']['metrics_by_iou']['0.33']['tp'], 1)
        self.assertEqual(r['metrics']['metrics_by_iou']['0.50']['tp'], 1)
        self.assertEqual(len(r['dataset_fingerprint']), 64)
        self.assertEqual(len(r['code_sha256']), 5)
        self.assertIn('opencv', r['environment'])

    def test_reject_unpinned(self):
        with self.assertRaisesRegex(ValueError, 'pinned'):
            benchmark(self.root, self.split)

    def test_fingerprint_changes(self):
        before = snapshot(self.root, self.raw)
        (self.root/'one.txt').write_text('8 8 17 16 1\n')
        self.assertNotEqual(before, snapshot(self.root, self.raw))

    def test_count(self):
        with self.assertRaisesRegex(ValueError, 'count'):
            benchmark(self.root, self.split, git_blob(self.raw), 2)


if __name__ == '__main__':
    unittest.main()
