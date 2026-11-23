"""Deterministic synthetic characterization of the current PCB differencing baseline."""
import json
import cv2
import numpy as np
from pcb_reference import detect


def make_reference():
    image = np.full((128, 128), 255, dtype=np.uint8)
    cv2.rectangle(image, (15, 20), (105, 26), 0, -1)
    cv2.rectangle(image, (45, 26), (51, 100), 0, -1)
    cv2.circle(image, (48, 100), 9, 0, -1)
    cv2.rectangle(image, (80, 60), (112, 66), 0, -1)
    return image


def run():
    rng = np.random.default_rng(20261123)
    ref = make_reference()
    cases = {'identical': ref.copy()}
    x = ref.copy(); x[40:44, 70:74] = 0; cases['defect_4x4'] = x
    x = ref.copy(); x[40:46, 70:76] = 0; cases['defect_6x6'] = x
    x = np.clip(ref.astype(np.int16)+25, 0, 255).astype(np.uint8); cases['brightness_plus25'] = x
    x = np.clip(ref.astype(np.int16)+40, 0, 255).astype(np.uint8); cases['brightness_plus40'] = x
    x = cv2.warpAffine(ref, np.float32([[1,0,1],[0,1,0]]), (128,128), borderValue=255); cases['translation_1px'] = x
    x = np.clip(ref.astype(np.float32)+rng.normal(0,10,ref.shape),0,255).astype(np.uint8); cases['noise_sigma10'] = x
    x = np.clip(ref.astype(np.float32)+rng.normal(0,20,ref.shape),0,255).astype(np.uint8); cases['noise_sigma20'] = x
    cases['blur_3x3_sigma0.8'] = cv2.GaussianBlur(ref,(3,3),.8)
    result = {}
    for name, image in cases.items():
        boxes = [list(row[1]) for row in detect(ref, image)]
        area = sum((b[2]-b[0])*(b[3]-b[1]) for b in boxes)
        result[name] = {'detections': len(boxes), 'boxes': boxes,
                        'sum_bbox_area': area, 'image_area_fraction': area/(128*128)}
    return {'purpose': 'synthetic failure-mode characterization only',
            'detector_parameters': {'pixel_threshold':35,'min_area':20,'connectivity':8},
            'cases': result}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, allow_nan=False))
