"""Class-aware, one-to-one detection evaluation at a fixed IoU threshold."""
from math import isfinite


def _box(box):
    if len(box) != 4 or not all(isfinite(x) for x in box):
        raise ValueError('Expected four finite xyxy coordinates')
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError('Box must have positive area')
    return box


def iou(a, b):
    a, b = _box(a), _box(b)
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return inter / ((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def evaluate_image(truth, predictions, threshold=.5):
    """Truth: (class, box); predictions: (class, box, score)."""
    if not 0 < threshold <= 1:
        raise ValueError('Invalid IoU threshold')
    truth, predictions = list(truth), list(predictions)
    for _, box in truth:
        _box(box)
    for _, box, score in predictions:
        _box(box)
        if not isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid confidence score')
    remaining = set(range(len(truth)))
    tp = 0
    for label, box, score in sorted(predictions, key=lambda p: -p[2]):
        candidates = [(iou(box, truth[i][1]), -i) for i in remaining if truth[i][0] == label]
        if candidates:
            overlap, neg_index = max(candidates)
            if overlap >= threshold:
                remaining.remove(-neg_index)
                tp += 1
    fp, fn = len(predictions)-tp, len(truth)-tp
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp/(tp+fp) if tp+fp else None,
            'recall': tp/(tp+fn) if tp+fn else None}
