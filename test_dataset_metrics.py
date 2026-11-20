import unittest
from dataset_metrics import average_precision

B = (0, 0, 10, 10)
C = (2, 0, 12, 10)

class DatasetTests(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(average_precision({'a': [(0,B)]}, {'a': [(0,B,.9)]})['mean_ap'], 1)
    def test_missing(self):
        self.assertEqual(average_precision({'a': [(0,B)]}, {})['mean_ap'], 0)
    def test_iou_threshold(self):
        self.assertEqual(average_precision({'a': [(0,B)]}, {'a': [(0,C,.9)]})['ap_by_iou'], {.5:1, .75:0})
    def test_negative_image(self):
        self.assertEqual(average_precision({'a': [(0,B)], 'b': []}, {'a': [(0,B,.8)], 'b': [(0,B,.9)]})['mean_ap'], .5)
    def test_class_average(self):
        self.assertEqual(average_precision({'a': [(0,B),(1,B)]}, {'a': [(0,B,.9)]})['mean_ap'], .5)
    def test_duplicate_rank(self):
        self.assertAlmostEqual(average_precision({'a': [(0,B)], 'b': [(0,B)]}, {'a': [(0,B,.9),(0,B,.8)], 'b':[(0,B,.7)]})['mean_ap'], (51+50*2/3)/101)
    def test_unknown_image(self):
        with self.assertRaises(ValueError):
            average_precision({'a': [(0,B)]}, {'b': []})
    def test_invalid_score(self):
        with self.assertRaises(ValueError):
            average_precision({'a': [(0,B)]}, {'a': [(0,B,float('nan'))]})
    def test_no_positive(self):
        with self.assertRaises(ValueError):
            average_precision({'a': []}, {})
