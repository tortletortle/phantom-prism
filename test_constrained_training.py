import unittest
import numpy as np
from constrained_training import frontier, select, split, noisy

class Constraints(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(strength=0., clean=.95, noisy=.70),
                     dict(strength=.05, clean=.94, noisy=.80),
                     dict(strength=.1, clean=.93, noisy=.90),
                     dict(strength=.2, clean=.92, noisy=.75)]
    def test_frontier(self):
        self.assertEqual([r['strength'] for r in frontier(self.rows)], [0., .05, .1])
    def test_constraint(self):
        self.assertEqual(select(self.rows), .05)
    def test_baseline(self):
        self.assertEqual(select(self.rows, 0.), 0.)
    def test_tie(self):
        self.assertEqual(select([self.rows[0], dict(strength=.1, clean=.95, noisy=.70)]), 0.)
    def test_partition(self):
        parts, _ = split()
        self.assertEqual([len(parts[k][1]) for k in ('train','validation','test')], [330,132,198])
        self.assertTrue(all(x.shape[1] == 256 for x,y in parts.values()))
    def test_noise_zero(self):
        x = np.array([0.,.5,1.])
        np.testing.assert_array_equal(x, noisy(x, 0., 1))

if __name__ == '__main__':
    unittest.main()
