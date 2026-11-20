"""Validate local benign inspection data before evaluation; never runs a detector."""
import hashlib
import json
import math
from pathlib import Path
from PIL import Image

SPLITS = ('train', 'validation', 'test')


def audit(root, rows):
    root = Path(root).resolve()
    scenes, digests, ids = {}, {}, set()
    counts = {s: {'images': 0, 'objects': 0, 'scenes': set()} for s in SPLITS}
    for row in rows:
        key, split, scene = (row[k] for k in ('id', 'split', 'scene'))
        if not all(isinstance(v, str) and v.strip() for v in (key, split, scene)):
            raise ValueError('Nonempty string id, split and scene required')
        if key in ids or split not in counts:
            raise ValueError('Duplicate id or invalid split')
        ids.add(key)
        if scenes.setdefault(scene, split) != split:
            raise ValueError('Scene leakage across splits')
        p = (root / row['image']).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            raise ValueError('Missing image or path outside dataset')
        with Image.open(p) as im:
            pixels = im.convert('RGB')
            w, h = pixels.size
            digest = hashlib.sha256(f'{w}x{h}:'.encode() + pixels.tobytes()).hexdigest()
        if digests.setdefault(digest, split) != split:
            raise ValueError('Duplicate decoded image across splits')
        for obj in row['objects']:
            box = obj['box']
            if not isinstance(obj['label'], str) or not obj['label'].strip():
                raise ValueError('Invalid label')
            if len(box) != 4 or not all(type(v) in (int, float) and math.isfinite(v) for v in box):
                raise ValueError('Invalid box coordinates')
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise ValueError('Box outside image or empty')
        counts[split]['images'] += 1
        counts[split]['objects'] += len(row['objects'])
        counts[split]['scenes'].add(scene)
    for v in counts.values():
        v['scenes'] = len(v['scenes'])
        if not v['images'] or not v['objects']:
            raise ValueError('Each split requires images and annotated positives')
    return counts


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.manifest.parent, json.loads(args.manifest.read_text(encoding='utf-8'))), indent=2))
