"""Validation-constrained benign noise training on three source photos.
Source-photo patch classification, not semantic detection or scene generalization.
"""
import hashlib
import json
import platform
import numpy as np
import sklearn
import skimage
from skimage import data
from sklearn.svm import SVC

NAMES = ('moon', 'coins', 'page')
STRENGTHS = (0., .05, .1, .2)
SEEDS = (901, 902, 903)
MARGIN = .01
MIN_GAIN = .01


def split():
    parts = {k: ([], []) for k in ('train', 'validation', 'test')}
    hashes = {}
    for label, name in enumerate(NAMES):
        image = getattr(data, name)()
        hashes[name] = hashlib.sha256(image.tobytes()).hexdigest()
        if image.ndim != 2 or image.shape[0] < 176 or image.shape[1] < 384:
            raise ValueError('Unexpected image shape')
        for part, lo, hi in [('train', 0, 160), ('validation', 192, 256), ('test', 288, 384)]:
            xs, ys = parts[part]
            for r in range(0, 176, 16):
                for c in range(lo, hi, 16):
                    xs.append(image[r:r+16, c:c+16].reshape(-1)/255.)
                    ys.append(label)
    return {k: (np.asarray(x), np.asarray(y)) for k, (x,y) in parts.items()}, hashes


def noisy(x, sigma, seed):
    return np.clip(x + np.random.default_rng(seed).normal(0, sigma, x.shape), 0, 1)


def frontier(rows):
    """Return nondominated candidates (maximize clean and noisy validation accuracy)."""
    return [a for a in rows if not any(b['clean'] >= a['clean'] and b['noisy'] >= a['noisy']
        and (b['clean'] > a['clean'] or b['noisy'] > a['noisy']) for b in rows)]


def select(rows, margin=MARGIN):
    baseline, = [r for r in rows if r['strength'] == 0]
    eligible = [r for r in rows if r['clean'] >= baseline['clean'] - margin - 1e-12]
    return max(eligible, key=lambda r: (r['noisy'], r['clean'], -r['strength']))['strength']


def counts(model, x, xn, y):
    return [int(np.sum(model.predict(v) == y)) for v in (x, xn)]


def run():
    parts, hashes = split()
    x, y = parts['train']; v, vy = parts['validation']; t, ty = parts['test']
    vn = noisy(v, .15, 910); tn = noisy(t, .15, 911)
    models, validation = {}, []
    for strength in STRENGTHS:
        raw = []
        for seed in SEEDS:
            # Same training budget and common random noise across strengths.
            model = SVC(C=3., gamma='scale').fit(np.concatenate([x, noisy(x, strength, seed)]), np.tile(y, 2))
            models[strength, seed] = model
            raw.append(counts(model, v, vn, vy))
        clean, degraded = np.mean(raw, axis=0)/len(vy)
        validation.append(dict(strength=strength, clean=float(clean), noisy=float(degraded), counts=raw))
    selected = select(validation)
    # Only the frozen selection and original baseline see test labels.
    test = []
    for seed in SEEDS:
        b = counts(models[0., seed], t, tn, ty)
        c = counts(models[selected, seed], t, tn, ty)
        test.append(dict(seed=seed, baseline=b, selected=c))
    delta = np.mean([np.subtract(r['selected'], r['baseline']) for r in test], axis=0)/len(ty)
    passed = bool(delta[0] >= -MARGIN and delta[1] >= MIN_GAIN and all(r['selected'][1] > r['baseline'][1] for r in test))
    return dict(scope='source-photo classification only', python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__, skimage=skimage.__version__, source_hashes=hashes,
                sizes={k: len(y) for k, (_,y) in parts.items()}, validation=validation,
                frontier=[r['strength'] for r in frontier(validation)], selected=selected, test=test,
                test_delta=delta.tolist(), accepted=passed)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, allow_nan=False))
