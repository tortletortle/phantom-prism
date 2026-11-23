"""Conservative translation-only ECC registration for paired grayscale PCB images."""
import cv2
import numpy as np


def align_translation(reference, inspected, max_shift=3.0, min_correlation=0.90,
                      iterations=200, epsilon=1e-7, gaussian_filter=5):
    if reference.shape != inspected.shape or reference.ndim != 2:
        raise ValueError('Expected equally sized grayscale images')
    if reference.dtype != np.uint8 or inspected.dtype != np.uint8:
        raise ValueError('Expected uint8 images')
    if not np.isfinite(max_shift) or max_shift <= 0:
        raise ValueError('max_shift must be positive and finite')
    if not np.isfinite(min_correlation) or not 0 < min_correlation <= 1:
        raise ValueError('min_correlation must be in (0, 1]')
    if iterations < 1 or epsilon <= 0 or gaussian_filter < 1 or gaussian_filter % 2 == 0:
        raise ValueError('Invalid ECC convergence parameters')
    template = reference.astype(np.float32) / 255.0
    moving = inspected.astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, epsilon)
    try:
        correlation, warp = cv2.findTransformECC(template, moving, warp,
            cv2.MOTION_TRANSLATION, criteria, None, gaussian_filter)
    except cv2.error as exc:
        raise ValueError('ECC registration failed') from exc
    dx, dy = float(warp[0, 2]), float(warp[1, 2])
    if not np.isfinite(correlation) or correlation < min_correlation:
        raise ValueError('ECC correlation below acceptance threshold')
    if max(abs(dx), abs(dy)) > max_shift:
        raise ValueError('Estimated translation exceeds max_shift')
    h, w = reference.shape
    aligned = cv2.warpAffine(inspected, warp, (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE)
    return aligned, {'correlation': float(correlation), 'dx': dx, 'dy': dy}
