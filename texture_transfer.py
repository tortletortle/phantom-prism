"""Spatially held-out real texture patches; not a detector or scene-independent benchmark."""
import hashlib
import json
import numpy as np
import skimage
from skimage import data
import torch
from torch import nn


def split():
    train, test, yt, yv, hashes = [], [], [], [], {}
    for label, name in enumerate(('brick', 'grass', 'gravel')):
        image = getattr(data, name)()
        hashes[name] = hashlib.sha256(image.tobytes()).hexdigest()
        for row in range(0, 512, 16):
            for col in range(0, 512, 16):
                tile = image[row:row+16, col:col+16][None] / 255.
                if col < 320:
                    train.append(tile); yt.append(label)
                elif col >= 384:
                    test.append(tile); yv.append(label)
    return (np.array(train, np.float32), np.array(yt, np.int64),
            np.array(test, np.float32), np.array(yv, np.int64), hashes)


def noise(x, rng, sigma):
    return np.clip(x + rng.normal(0, sigma, x.shape), 0, 1).astype(np.float32)


def fit(x, y, seed):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 3))
    opt = torch.optim.Adam(model.parameters(), lr=.003)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    rng = torch.Generator().manual_seed(seed+1000)
    for _ in range(12):
        for idx in torch.randperm(len(y), generator=rng).split(128):
            opt.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(model(x[idx]), y[idx]).backward()
            opt.step()
    return model.eval()


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    x, y, test, labels, hashes = split()
    noisy = noise(test, np.random.default_rng(800), .15)
    rows = []
    for seed in (801, 802, 803):
        rng = np.random.default_rng(seed+10)
        mixed = noise(x, rng, rng.choice([0., .1, .2], (len(x), 1, 1, 1)))
        for arm, second in [('clean_repeat', x), ('retain_clean', mixed)]:
            model = fit(np.concatenate([x, second]), np.tile(y, 2), seed)
            with torch.no_grad():
                counts = [int((model(torch.from_numpy(v)).argmax(1).numpy() == labels).sum()) for v in (test, noisy)]
            rows.append(dict(seed=seed, arm=arm, clean_correct=counts[0], noisy_correct=counts[1]))
    pairs = [(rows[i], rows[i+1]) for i in range(0, len(rows), 2)]
    clean_delta = np.mean([b['clean_correct']-a['clean_correct'] for a,b in pairs])/len(labels)
    noisy_delta = np.mean([b['noisy_correct']-a['noisy_correct'] for a,b in pairs])/len(labels)
    passed = bool(clean_delta >= -.01 and noisy_delta >= .01 and all(b['noisy_correct'] > a['noisy_correct'] for a,b in pairs))
    return dict(numpy=np.__version__, skimage=skimage.__version__, torch=torch.__version__,
                source_sha256=hashes, train_patches=len(y), test_patches=len(labels), rows=rows,
                clean_delta=float(clean_delta), noisy_delta=float(noisy_delta), acceptance_passed=passed)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
