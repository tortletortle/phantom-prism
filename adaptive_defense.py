"""Quality-gated denoising for synthetic industrial inspection, not real cameras."""
import json
import numpy as np
from scipy.ndimage import median_filter
from sklearn.linear_model import LogisticRegression
from inspection_demo import SIDE, GRID, samples, cell_box, summarize

CELLS = GRID**2


def make_split(seed, n):
    rng = np.random.default_rng(seed)
    tiles, labels = samples(rng, n*CELLS)
    # Fixed mixture of clean frames and ordinary sensor noise; no search.
    levels = rng.choice([0., .06, .12, .18], n)
    x = np.clip(tiles + rng.normal(size=tiles.shape)*np.repeat(levels, CELLS)[:, None, None], 0, 1)
    return x, labels.reshape(n, CELLS), levels == 0


def views(model, tiles):
    smooth = median_filter(tiles, size=(1, 3, 3))
    quality = np.median(np.abs(tiles-smooth).reshape(-1, CELLS*SIDE*SIDE), axis=1)
    def score(x):
        return model.predict_proba(x.reshape(len(x), -1))[:, 1].reshape(-1, CELLS)
    return quality, score(tiles), score(smooth)


def gate(quality, raw, filtered, cutoff):
    return np.where((quality > cutoff)[:, None], filtered, raw)


def choose_cutoff(quality, raw, filtered, labels):
    # Calibration only: minimize cell error, ties prefer less intervention.
    candidates = [float('inf'), *sorted(set(np.quantile(quality, np.linspace(0, 1, 21))), reverse=True), -float('inf')]
    return min(candidates, key=lambda t: np.mean((gate(quality, raw, filtered, t) >= .5) != labels))


def evaluate(labels, scores):
    truth = {str(i): [(0, cell_box(j)) for j, y in enumerate(row) if y] for i, row in enumerate(labels)}
    pred = {str(i): [(0, cell_box(j), float(s)) for j, s in enumerate(row)] for i, row in enumerate(scores)}
    return summarize(truth, pred)


def paired_ci(delta, seed=501):
    # Independent synthetic images are the resampling unit, not tiles.
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(len(delta), size=(2000, len(delta)))].mean(axis=1)
    return [float(v) for v in np.quantile(boot, [.025, .975])]


def run():
    train, y = samples(np.random.default_rng(100), 1800)
    model = LogisticRegression(max_iter=500, random_state=100).fit(train.reshape(len(train), -1), y)
    x, labels, _ = make_split(310, 160)
    q, raw, filtered = views(model, x)
    cutoff = choose_cutoff(q, raw, filtered, labels)
    rows, differences = [], []
    for seed in range(401, 406):
        x, labels, clean = make_split(seed, 120)
        q, raw, filtered = views(model, x)
        adaptive = gate(q, raw, filtered, cutoff)
        scores = {'raw': raw, 'fixed': filtered, 'adaptive': adaptive}
        rows.append({'seed': seed, 'images': len(labels), 'filtered_fraction': float(np.mean(q > cutoff)),
                     'all': {k: evaluate(labels, v) for k, v in scores.items()},
                     'clean': {k: evaluate(labels[clean], v[clean]) for k, v in scores.items()}})
        for name in ('raw', 'fixed'):
            err = ((adaptive >= .5) != labels).mean(axis=1) - ((scores[name] >= .5) != labels).mean(axis=1)
            differences.append((name, err))
    comparisons = {}
    for name in ('raw', 'fixed'):
        d = np.concatenate([v for k, v in differences if k == name])
        comparisons[name] = {'error_delta': float(d.mean()), 'ci95': paired_ci(d)}
    return {'scope': 'synthetic fixed-grid inspection', 'calibration_seed': 310, 'calibration_images': 160,
            'cutoff': str(cutoff), 'rows': rows, 'adaptive_minus_baseline_cell_error': comparisons}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, allow_nan=False))
