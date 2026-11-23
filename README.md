# Phantom Prism / 幻影棱镜

物理成像条件下的视觉检测鲁棒性研究。当前优先做**防御性评估、可复现实验和失效分析**；不把合成实验或单张样本演示当成真实世界鲁棒性证据。

## Current stage: pre-hardware validation

**Not a V1.0 release. No physical camera test has been completed.**

### Implemented
- Paired-image PCB difference baseline: `pcb_reference.py`.
- Dataset path, annotation and split provenance checks: `inspection_preflight.py`, `deep_pcb_import.py`, `pcb_benchmark.py`.
- Class-agnostic localization counts at IoU 0.33 and 0.50.
- Deterministic synthetic failure-mode characterization: `prehardware_stress.py`.
- Regression tests for file validation, detector behavior and known failure modes.

### Important limits
- The current PCB baseline only localizes changed regions. It does **not** classify the six DeepPCB defect categories and cannot report official DeepPCB mAP/F-score.
- The DeepPCB README specifies class-matched evaluation, IoU strictly greater than 0.33, AP and F-score. Internal class-agnostic localization metrics are not directly comparable.
- The baseline uses fixed absolute grayscale difference (pixel threshold 35), 8-connected components and minimum area 20 pixels. It is sensitive to image registration, blur and sufficiently large illumination shifts; tiny regions can be discarded.
- The full DeepPCB corpus has not been transferred into the current runtime. One source image pair and its annotation were inspected through GitHub metadata access, but not locally decoded or benchmarked.
- No YOLO/CNN detector has been trained or evaluated on DeepPCB. No physical camera, printed target, or hardware test has been performed.

## Pre-hardware gate

1. Obtain the complete research-use dataset and preserve its original split and file hashes.
2. Reproduce the class-aware official evaluation protocol (or use the upstream evaluator) for any six-class model. Keep the class-agnostic reference baseline clearly separate.
3. Run the full offline dataset benchmark and inspect per-image false positives and misses.
4. Test registration, illumination, blur, sensor noise and small-defect sensitivity on held-out validation data. Do not tune on the official test set.
5. Only after these gates pass, perform a controlled camera test on owned/authorized PCB samples with fixed camera geometry and documented lighting.

## Reproduction

Run the synthetic characterization:

```bash
python prehardware_stress.py
python -m unittest -v
```

Run the fixed class-agnostic reference baseline after obtaining the original DeepPCB data:

```bash
python pcb_benchmark.py path/to/DeepPCB/PCBData pcb_results.json
```

The benchmark entry point verifies the pinned official test-list Git blob and fingerprints the referenced image pairs and annotations. Its output is a localization-only engineering baseline, not an official DeepPCB score.

## Research integrity

Every result must identify whether it is a unit test, synthetic experiment, public-dataset result, or physical-camera result. Report negative results and known limitations. Do not describe a feature as validated until the corresponding evidence exists.
