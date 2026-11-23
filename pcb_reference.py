"""Fixed, class-agnostic reference-image baseline for aligned PCB localization."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from deep_pcb_import import local_file, parse_annotation
from detection_metrics import evaluate_image

IOU_THRESHOLDS = (0.33, 0.50)


def detect(reference, inspected, pixel_threshold=35, min_area=20):
    if reference.shape != inspected.shape or reference.ndim != 2:
        raise ValueError('Expected equally sized aligned grayscale images')
    if reference.dtype != np.uint8 or inspected.dtype != np.uint8:
        raise ValueError('Expected uint8 images')
    if type(pixel_threshold) is not int or not 0 <= pixel_threshold <= 255:
        raise ValueError('pixel_threshold must be an integer in [0, 255]')
    if type(min_area) is not int or min_area < 1:
        raise ValueError('min_area must be a positive integer')
    delta = cv2.absdiff(reference, inspected)
    mask = (delta >= pixel_threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = []
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if area >= min_area:
            result.append(('defect', (x, y, x+w, y+h), float(delta[labels == i].mean()/255)))
    return result


def read_gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('L')).copy()


def _scores(tp, fp, fn):
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/(tp+fn) if tp+fn else None
    f1 = 2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def run(root, split_file, pixel_threshold=35, min_area=20):
    root = Path(root).resolve()
    sums = {t: dict(tp=0, fp=0, fn=0) for t in IOU_THRESHOLDS}
    ids, images_with_predictions, images_with_gt = set(), 0, 0
    for number, line in enumerate(Path(split_file).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2 or fields[0] in ids:
            raise ValueError(f'Invalid or duplicate split row {number}')
        ids.add(fields[0])
        p = Path(fields[0])
        if p.suffix.lower() != '.jpg':
            raise ValueError(f'Expected original jpg path at row {number}')
        reference = read_gray(local_file(root, p.with_name(p.stem+'_temp.jpg')))
        inspected = read_gray(local_file(root, p.with_name(p.stem+'_test.jpg')))
        h, w = inspected.shape
        truth = []
        for obj in parse_annotation(local_file(root, fields[1]).read_text(encoding='utf-8-sig')):
            x1, y1, x2, y2 = obj['box']
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise ValueError(f'Invalid annotation box at row {number}')
            truth.append(('defect', obj['box']))
        predictions = detect(reference, inspected, pixel_threshold, min_area)
        images_with_predictions += bool(predictions)
        images_with_gt += bool(truth)
        for threshold in IOU_THRESHOLDS:
            counts = evaluate_image(truth, predictions, threshold=threshold)
            for key in ('tp', 'fp', 'fn'):
                sums[threshold][key] += counts[key]
    if not ids:
        raise ValueError('Empty split')
    return dict(scope='class-agnostic localization only; not official DeepPCB mAP/F-score',
                images=len(ids), images_with_predictions=images_with_predictions,
                images_with_ground_truth=images_with_gt,
                metrics_by_iou={f'{t:.2f}': _scores(**sums[t]) for t in IOU_THRESHOLDS},
                pixel_threshold=pixel_threshold, min_area=min_area,
                opencv=cv2.__version__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Original PCBData directory')
    parser.add_argument('split', type=Path, help='Original file list')
    parser.add_argument('--pixel-threshold', type=int, default=35)
    parser.add_argument('--min-area', type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.split, args.pixel_threshold, args.min_area), indent=2, allow_nan=False))
