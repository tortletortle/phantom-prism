"""Matched-budget benign noise training on recorded digits; classification only."""
import json
import platform
import numpy as np
import sklearn
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn


def make_arms(x, y, rng):
    sigma = rng.choice([0., .1, .2], size=(len(x), 1, 1, 1))
    mixed = np.clip(x + rng.normal(0, sigma, x.shape), 0, 1).astype(np.float32)
    return {'clean_repeat': np.concatenate([x, x]),
            'mixed_repeat': np.concatenate([mixed, mixed]),
            'retain_clean': np.concatenate([x, mixed])}, np.tile(y, 2)


def model_for(name):
    if name == 'linear':
        return nn.Sequential(nn.Flatten(), nn.Linear(64, 10))
    if name == 'cnn':
        return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                             nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(),
                             nn.MaxPool2d(2), nn.Flatten(), nn.Linear(24*4*4, 10))
    raise ValueError(name)


def fit(name, x, y, seed):
    torch.manual_seed(seed)
    m = model_for(name)
    opt = torch.optim.Adam(m.parameters(), lr=.003)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    order = torch.Generator().manual_seed(seed+1000)
    for _ in range(20):
        for idx in torch.randperm(len(y), generator=order).split(64):
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(m(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    return m.eval()


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data = load_digits()
    x = (data.images[:, None]/16.).astype(np.float32)
    y = data.target.astype(np.int64)
    rows = []
    for seed in (701, 702, 703):
        train, test = train_test_split(np.arange(len(y)), test_size=.25,
                                       stratify=y, random_state=seed)
        arms, labels = make_arms(x[train], y[train], np.random.default_rng(seed+10))
        noisy = np.clip(x[test]+np.random.default_rng(seed+20).normal(0, .15, x[test].shape), 0, 1).astype(np.float32)
        for name in ('linear', 'cnn'):
            for arm, images in arms.items():
                m = fit(name, images, labels, seed)
                with torch.no_grad():
                    clean_count = int((m(torch.from_numpy(x[test])).argmax(1).numpy()==y[test]).sum())
                    noisy_count = int((m(torch.from_numpy(noisy)).argmax(1).numpy()==y[test]).sum())
                rows.append(dict(seed=seed, model=name, arm=arm, clean_correct=clean_count,
                                 noisy_correct=noisy_count, test_count=len(test), train_views=len(labels)))
    return dict(scope='recorded digit classification only', python=platform.python_version(),
                numpy=np.__version__, sklearn=sklearn.__version__, torch=torch.__version__, rows=rows)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
