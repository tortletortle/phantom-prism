"""Paired evaluation of detection recall under benign imaging variations."""
import math
import random
from statistics import mean


def paired_recall(clean, changed, seed=42, resamples=2000):
    """Inputs are aligned per-image recall values from annotated test images."""
    clean, changed = list(clean), list(changed)
    if not clean or len(clean) != len(changed):
        raise ValueError('Nonempty aligned samples required')
    if resamples < 100:
        raise ValueError('At least 100 bootstrap resamples required')
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in clean + changed):
        raise ValueError('Recall must be finite and within [0, 1]')
    delta = [b-a for a,b in zip(clean, changed)]
    rng = random.Random(seed)
    n = len(delta)
    boot = sorted(mean(rng.choices(delta, k=n)) for _ in range(resamples))
    return {'n': n, 'clean_macro_recall': mean(clean),
            'changed_macro_recall': mean(changed), 'paired_delta': mean(delta),
            'bootstrap_ci95': (boot[int(.025*(resamples-1))], boot[int(.975*(resamples-1))]),
            'seed': seed, 'resamples': resamples}
