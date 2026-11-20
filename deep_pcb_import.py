"""Import locally obtained DeepPCB research annotations with explicit board groups."""
import argparse
import csv
import json
import re
from pathlib import Path
from inspection_preflight import audit

LABELS = {1: 'open', 2: 'short', 3: 'mousebite', 4: 'spur', 5: 'copper', 6: 'pin-hole'}


def local_file(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Missing or out-of-root file: {name}')
    return path


def parse_annotation(text):
    objects = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            x1, y1, x2, y2, cls = map(int, re.split(r'[,\s]+', line.strip()))
            label = LABELS[cls]
        except (ValueError, KeyError) as exc:
            raise ValueError(f'Invalid annotation at line {number}') from exc
        objects.append(dict(label=label, box=[x1, y1, x2, y2]))
    return objects


def convert(index):
    index = Path(index).resolve()
    root = index.parent
    rows = []
    with index.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'id', 'split', 'board_id', 'image', 'annotation'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('Missing required index columns')
        for row in reader:
            annotation = local_file(root, row['annotation'])
            rows.append(dict(id=row['id'], split=row['split'], scene=row['board_id'],
                             image=row['image'], objects=parse_annotation(annotation.read_text(encoding='utf-8-sig'))))
    return rows, audit(root, rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=Path)
    args = parser.parse_args()
    rows, counts = convert(args.index)
    out = args.index.resolve().parent / 'manifest.json'
    with out.open('x', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(json.dumps(counts, indent=2))
