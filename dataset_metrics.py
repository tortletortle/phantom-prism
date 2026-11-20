"""Offline reference AP for ordinary annotated images (no crowd/ignore handling)."""
from math import isfinite
from detection_metrics import iou


def average_precision(truth, predictions, thresholds=(.5, .75)):
    """Mappings image_id -> [(class, xyxy)] and [(class, xyxy, score)].

    Uses 101-point interpolated AP. Only classes with ground truth are scored.
    All predictions must refer to a declared image, including negative images.
    Not a replacement for official COCOeval (no crowd, area or maxDets rules).
    """
    thresholds = tuple(thresholds)
    if not thresholds or len(set(thresholds)) != len(thresholds) or any(not 0 < t <= 1 for t in thresholds):
        raise ValueError('Expected distinct IoU thresholds in (0, 1]')
    if not set(predictions) <= set(truth):
        raise ValueError('Unknown image ID')
    truth = {k: list(v) for k, v in truth.items()}
    predictions = {k: list(v) for k, v in predictions.items()}
    classes = {label for rows in truth.values() for label, _ in rows}
    if not classes:
        raise ValueError('No annotated positives')
    for rows in truth.values():
        for _, box in rows:
            iou(box, box)
    for rows in predictions.values():
        for _, box, score in rows:
            iou(box, box)
            if not isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid confidence score')
    by_threshold = {}
    for threshold in thresholds:
        by_class = []
        for label in classes:
            targets = {k: [b for c, b in rows if c == label] for k, rows in truth.items()}
            count = sum(map(len, targets.values()))
            # Stable ties follow declared image and prediction order.
            ranked = sorted(((k, b, s) for k in truth for c, b, s in predictions.get(k, ()) if c == label), key=lambda x: -x[2])
            matched = {k: set() for k in targets}
            curve = []
            tp = 0
            for rank, (key, box, _) in enumerate(ranked, 1):
                best, index = max(((iou(box, b), -j) for j, b in enumerate(targets[key]) if j not in matched[key]), default=(0, 0))
                if best >= threshold:
                    matched[key].add(-index)
                    tp += 1
                curve.append((tp/count, tp/rank))
            ap = sum(max((p for r,p in curve if r >= i/100), default=0) for i in range(101))/101
            by_class.append(ap)
        by_threshold[threshold] = sum(by_class)/len(by_class)
    return {'ap_by_iou': by_threshold, 'mean_ap': sum(by_threshold.values())/len(by_threshold), 'images': len(truth), 'classes': len(classes)}
