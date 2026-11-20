import unittest
from metrics import paired_recall

class TestMetrics(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(paired_recall([0, .5, 1], [0, .5, 1])['bootstrap_ci95'], (0, 0))
    def test_known_delta(self):
        r = paired_recall([1]*8, [.5]*8)
        self.assertEqual(r['paired_delta'], -.5)
        self.assertEqual(r['bootstrap_ci95'], (-.5, -.5))
    def test_seed(self):
        for seed in range(5):
            self.assertEqual(paired_recall([1, 0, 1], [0, 1, 1], seed), paired_recall([1, 0, 1], [0, 1, 1], seed))
    def test_invalid(self):
        for a,b in [([], []), ([1], [1,0]), ([float('nan')], [1]), ([1], [2])]:
            with self.assertRaises(ValueError):
                paired_recall(a,b)

if __name__ == '__main__':
    unittest.main()
