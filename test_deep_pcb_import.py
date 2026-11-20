import csv
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from deep_pcb_import import convert, parse_annotation


class ImportTests(unittest.TestCase):
    def test_delimiters_and_labels(self):
        rows = parse_annotation('1,2,3,4,1\n1 2 3 4 6\n')
        self.assertEqual([r['label'] for r in rows], ['open', 'pin-hole'])
        self.assertEqual(rows[1]['box'], [1, 2, 3, 4])

    def test_invalid_annotations(self):
        for value in ('1 2 3 4 0', '1 2 3 4 7', '1 2 3 4', '1.5 2 3 4 1'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'line 1'):
                parse_annotation(value)

    def test_empty_annotation(self):
        self.assertEqual(parse_annotation('\n'), [])

    def test_index(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            index = root/'index.csv'
            rows = []
            for i, split in enumerate(('train','validation','test')):
                Image.new('RGB', (16,16), (i*60,0,0)).save(root/f'{i}.png')
                (root/f'{i}.txt').write_text('1,1,8,8,2')
                rows.append(dict(id=str(i), split=split, board_id=f'board{i}', image=f'{i}.png', annotation=f'{i}.txt'))
            def write():
                with index.open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            write()
            manifest, counts = convert(index)
            self.assertEqual(counts['test']['objects'], 1)
            self.assertEqual(manifest[0]['objects'][0]['label'], 'short')
            rows[2]['board_id'] = rows[0]['board_id']
            write()
            with self.assertRaisesRegex(ValueError, 'Scene leakage'):
                convert(index)
            rows[2]['board_id'] = 'board2'
            rows[2]['annotation'] = '../missing.txt'
            write()
            with self.assertRaisesRegex(ValueError, 'Missing or out-of-root'):
                convert(index)
