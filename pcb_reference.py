"""Fixed, class-agnostic reference-image baseline for aligned PCB inspection."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from deep_pcb_import import local_file, parse_annotation
from detection_metrics import evaluate_image


def detect(reference, inspected):
    if reference.shape != inspected.shape or reference.ndim != 2:
        raise ValueError('Expected equally sized aligned grayscale images')
    if reference.dtype != np.uint8 or inspected.dtype != np.uint8:
        raise ValueError('Expected uint8 images')
    delta = cv2.absdiff(reference, inspected)
    mask = (delta >= 35).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = []
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if area >= 20:
            result.append(('defect', (x, y, x+w, y+h), float(delta[labels == i].mean()/255)))
    return result


def read_gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('L')).copy()


def run(root, split_file):
    root = Path(root).resolve()
    sums = dict(tp=0, fp=0, fn=0)
    ids = set()
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
        counts = evaluate_image(truth, detect(reference, inspected), threshold=.5)
        for k in sums:
            sums[k] += counts[k]
    if not ids:
        raise ValueError('Empty split')
    tp, fp, fn = (sums[k] for k in ('tp', 'fp', 'fn'))
    return dict(scope='class-agnostic aligned PCB baseline', images=len(ids), **sums,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                threshold=35, min_area=20, iou=.5, opencv=cv2.__version__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Original PCBData directory')
    parser.add_argument('split', type=Path, help='Original file list')
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.split), indent=2, allow_nan=False))
