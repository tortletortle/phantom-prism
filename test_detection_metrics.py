import unittest
from detection_metrics import iou, evaluate_image

B = (0, 0, 10, 10)

class DetectionTests(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(iou(B, B), 1)
        self.assertEqual(iou(B, (10, 10, 20, 20)), 0)
        self.assertAlmostEqual(iou(B, (5, 0, 15, 10)), 1/3)
    def test_duplicate(self):
        self.assertEqual(evaluate_image([(0, B)], [(0, B, .9), (0, B, .8)]),
                         dict(tp=1, fp=1, fn=0, precision=.5, recall=1))
    def test_wrong_class(self):
        self.assertEqual(evaluate_image([(0, B)], [(1, B, .9)]),
                         dict(tp=0, fp=1, fn=1, precision=0, recall=0))
    def test_empty(self):
        self.assertIsNone(evaluate_image([], [])['recall'])
        self.assertEqual(evaluate_image([(0, B)], [])['recall'], 0)
    def test_invalid(self):
        with self.assertRaises(ValueError):
            iou(B, (0, 0, 0, 1))
        with self.assertRaises(ValueError):
            evaluate_image([], [(0, B, float('nan'))])
    def test_low_overlap(self):
        self.assertEqual(evaluate_image([(0, B)], [(0, (5, 0, 15, 10), .9)])['tp'], 0)
