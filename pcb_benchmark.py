"""Provenance report for the fixed DeepPCB reference localization baseline."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import cv2
import numpy as np
from PIL import __version__ as pillow_version
from deep_pcb_import import local_file
from pcb_reference import run

OFFICIAL_TEST = '42b9a1dfc363182dbccc44cb2fd195f913ed87ea'


def git_blob(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


def snapshot(root, raw):
    digest = hashlib.sha256()
    count = 0
    ids = set()
    for line in raw.decode('utf-8-sig').splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError('Malformed official split row')
        image, annotation = fields
        if image in ids:
            raise ValueError('Duplicate image ID')
        ids.add(image)
        p = Path(image)
        for name in (p.with_name(p.stem+'_temp.jpg'), p.with_name(p.stem+'_test.jpg'), Path(annotation)):
            data = local_file(root, name).read_bytes()
            digest.update(json.dumps([name.as_posix(), hashlib.sha256(data).hexdigest()]).encode() + b'\n')
        count += 1
    return count, digest.hexdigest()


def benchmark(root, split, expected_blob=OFFICIAL_TEST, expected_count=500, pixel_threshold=35, min_area=20):
    root, split = Path(root).resolve(), Path(split).resolve()
    raw = split.read_bytes()
    if git_blob(raw) != expected_blob:
        raise ValueError('Split does not match pinned list')
    before = snapshot(root, raw)
    if before[0] != expected_count:
        raise ValueError('Unexpected image count')
    metrics = run(root, split, pixel_threshold, min_area)
    if split.read_bytes() != raw or snapshot(root, raw) != before or metrics['images'] != before[0]:
        raise ValueError('Inputs changed during evaluation')
    here = Path(__file__).resolve().parent
    code = {name: hashlib.sha256((here/name).read_bytes()).hexdigest() for name in (
        'pcb_benchmark.py', 'pcb_reference.py', 'deep_pcb_import.py', 'inspection_preflight.py', 'detection_metrics.py')}
    return dict(protocol='fixed class-agnostic localization baseline; not official class-aware benchmark',
                split_git_blob=expected_blob, dataset_fingerprint=before[1], code_sha256=code,
                environment=dict(python=platform.python_version(), numpy=np.__version__,
                                 opencv=cv2.__version__, pillow=pillow_version),
                metrics=metrics)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--pixel-threshold', type=int, default=35)
    ap.add_argument('--min-area', type=int, default=20)
    args = ap.parse_args()
    result = benchmark(args.root, args.root/'test.txt', pixel_threshold=args.pixel_threshold, min_area=args.min_area)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
