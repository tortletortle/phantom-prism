import unittest
import numpy as np
from adaptive_defense import gate, choose_cutoff, paired_ci, make_split

class AdaptiveTests(unittest.TestCase):
    def test_gate(self):
        r = np.array([[.1], [.2]])
        f = np.array([[.8], [.9]])
        np.testing.assert_array_equal(gate(np.array([.1, .3]), r, f, .2), [[.1], [.9]])
    def test_calibration_prefers_raw_when_equal(self):
        r = np.array([[.1], [.9]])
        self.assertEqual(choose_cutoff(np.array([.1, .2]), r, r, [[0], [1]]), float('inf'))
    def test_calibration_selects_filter(self):
        r = np.array([[.1], [.1]])
        f = np.array([[.9], [.9]])
        q = np.array([.1, .9])
        t = choose_cutoff(q, r, f, [[0], [1]])
        np.testing.assert_array_equal(gate(q, r, f, t) >= .5, [[False], [True]])
    def test_bootstrap_zero(self):
        self.assertEqual(paired_ci(np.zeros(20)), [0., 0.])
    def test_split_reproducible(self):
        for a,b in zip(make_split(310, 8), make_split(310, 8)):
            np.testing.assert_array_equal(a, b)
