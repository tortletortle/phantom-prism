"""Benign sensor-noise training benchmark on sklearn's recorded digit dataset.
Classification only; no detector, physical experiment or adversarial evaluation.
"""
import hashlib
import json
import platform
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def add_noise(x, sigma, rng):
    return np.clip(x + rng.normal(0, sigma, x.shape), 0, 1)


def run():
    data = load_digits()
    x, y = data.data / 16., data.target
    pool, test = train_test_split(np.arange(len(y)), test_size=.25,
                                 random_state=606, stratify=y)
    noisy_test = add_noise(x[test], .15, np.random.default_rng(607))
    rows = []
    for seed in range(610, 615):
        rng = np.random.default_rng(seed)
        train = rng.choice(pool, 1200, replace=False)
        sigma = rng.choice([0., .1, .2], size=(len(train), 1))
        augmented = add_noise(x[train], sigma, rng)
        for name, factory in [('logistic', lambda: LogisticRegression(max_iter=1000, random_state=seed)),
                              ('linear_svm', lambda: LinearSVC(random_state=seed, max_iter=5000))]:
            row = {'seed': seed, 'model': name}
            for arm, inputs in [('clean_training', x[train]), ('mixed_training', augmented)]:
                model = factory().fit(inputs, y[train])
                row[arm] = {'clean_correct': int((model.predict(x[test]) == y[test]).sum()),
                            'noise_correct': int((model.predict(noisy_test) == y[test]).sum())}
            rows.append(row)
    return {'scope': 'recorded digit classification only', 'python': platform.python_version(),
            'numpy': np.__version__, 'sklearn': sklearn.__version__,
            'dataset_sha256': hashlib.sha256(data.data.astype('<f8').tobytes() + y.astype('<i8').tobytes()).hexdigest(),
            'split_seed': 606, 'noise_seed': 607, 'train_count': 1200, 'test_count': len(test),
            'rows': rows}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
