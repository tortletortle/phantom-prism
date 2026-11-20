"""Synthetic industrial inspection smoke test, not a real-camera benchmark."""
import json
import numpy as np
from scipy.ndimage import median_filter
from sklearn.linear_model import LogisticRegression
from dataset_metrics import average_precision
from detection_metrics import evaluate_image

SIDE = 12
GRID = 3


def samples(rng, n):
    """Create independent grayscale inspection tiles and presence labels."""
    y = rng.integers(0, 2, n)
    yy, xx = np.mgrid[:SIDE, :SIDE]
    tiles = rng.uniform(.35, .65, (n, 1, 1)) + rng.normal(0, .035, (n, SIDE, SIDE))
    for i, label in enumerate(y):
        cx, cy = rng.uniform(4.5, 6.5, 2)
        radius = np.hypot(xx-cx, yy-cy)
        shape = (radius >= 2) & (radius <= 4) if label else radius <= 2
        tiles[i] += shape * rng.uniform(.08, .28)
    return np.clip(tiles, 0, 1), y


def cell_box(j):
    row, col = divmod(j, GRID)
    return (col*SIDE, row*SIDE, (col+1)*SIDE, (row+1)*SIDE)


def predict(model, tiles):
    scores = model.predict_proba(tiles.reshape(len(tiles), -1))[:, 1]
    return {str(i): [(0, cell_box(j), float(s)) for j, s in enumerate(row)]
            for i, row in enumerate(scores.reshape(-1, GRID**2))}


def summarize(truth, predictions):
    counts = [evaluate_image(truth[k], [p for p in predictions[k] if p[2] >= .5]) for k in truth]
    tp, fp, fn = (sum(x[k] for x in counts) for k in ('tp', 'fp', 'fn'))
    return {'ap50': average_precision(truth, predictions, (.5,))['mean_ap'],
            'precision': tp/(tp+fp) if tp+fp else None,
            'recall': tp/(tp+fn) if tp+fn else None,
            'tp': tp, 'fp': fp, 'fn': fn}


def run():
    train, labels = samples(np.random.default_rng(100), 1800)
    model = LogisticRegression(max_iter=500, random_state=100).fit(train.reshape(len(train), -1), labels)
    rows = []
    for seed in (201, 202, 203):
        rng = np.random.default_rng(seed)
        clean, y = samples(rng, 60*GRID**2)
        truth = {str(i): [(0, cell_box(j)) for j, label in enumerate(row) if label]
                 for i, row in enumerate(y.reshape(-1, GRID**2))}
        noisy = np.clip(clean + rng.normal(0, .12, clean.shape), 0, 1)
        views = {'clean': clean, 'sensor_noise': noisy,
                 'clean_denoised': median_filter(clean, size=(1, 3, 3)),
                 'noise_denoised': median_filter(noisy, size=(1, 3, 3))}
        rows.append({'seed': seed, 'images': 60, 'results': {k: summarize(truth, predict(model, v)) for k, v in views.items()}})
    return {'scope': 'synthetic tile inspection only', 'train_seed': 100, 'train_tiles': 1800,
            'noise_std': .12, 'confidence_threshold': .5, 'runs': rows}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, allow_nan=False))
