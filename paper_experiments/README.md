# Code and results: gated dual-CNN visibility pipeline for firefighters

Everything needed to reproduce the numbers in the paper, plus the artifacts deployed
to the Raspberry Pi. No hidden state: every reported figure traces to a script here
and a JSON in `results/`.

## Layout

```
config.py         paths, in one place (override with env vars)
experiments/      the scripts that produced every reported number
results/          their raw JSON output, as committed
deploy/           what actually runs on the Pi (models + benchmark)
superseded/       earlier runs NOT used in the paper, kept for transparency
```

## Setup

```bash
pip install tensorflow numpy opencv-python scikit-image scipy pillow matplotlib SciencePlots
export CNN_REPO=/path/to/dual-cnn-system-firefighters    # the models + BIPEDv2 + SMOKE
python3 experiments/eval_strict.py
```

`CNN_REPO` must contain, unchanged from the published repo:

| Path | Used for |
|---|---|
| `edge/saved_models/cnn_trial_13.keras` | the deployed 23K edge model |
| `edge/saved_models/cnn_trial_7.keras` | the 1.95M accuracy-oriented variant |
| `edge/lite/lite-models/model_10.tflite` | the original uint8 export |
| `edge/BIPEDv2/BIPED/edges/` | edge train/test images and annotations |
| `smoke/saved_models/dehazer_trial_8.keras` | the deployed dehazer |
| `smoke/SMOKE/{train,test}/{hazy,clean}/` | 165 train / 12 test real smoke pairs |

## Which script produced which claim

| Paper claim | Script | Result file |
|---|---|---|
| Edge ODS 0.738 / OIS 0.755 (deployed), 0.786 / 0.800 (1.95M), Canny 0.692 | `eval_strict.py` | `results_strict.json` |
| uint8 edge ODS 0.740, quantization is free | `eval_strict.py` | `results_strict.json` |
| Dehazer 18.60 dB, DCP 13.04, hazy input 13.60 | `eval_extras.py` | `results_extras.json` |
| AOD-Net 17.08 dB, DehazeNet 13.60 (identical 165-pair corpus) | `baselines_dehaze_fair.py` (AOD-Net), `baselines_dehazenet_165.py` (DehazeNet) | `results_baselines_fair.json`, `results/logs/` |
| Quantized dehazer 16.80 dB, −1.8 dB cost | `quantize_dehazer.py` | `results_quant_dehazer.json` |
| Crossover: dehazing helps only past a haze density | `eval_pipeline.py` | `results_pipeline.json` |
| **Gate: mean ODS 0.675 vs 0.664 / 0.630, tau 0.585, firing rates** | **`gate.py`** | **`results_gate.json`** |
| MACs 0.52 / 2.28 GMAC, thread invariance | `hw_check.py` | `results_hw.json` |
| Parameter counts, r=1 tolerance check | `fix_r1.py` | `results_fix.json` |
| TILE removal, op set 10 -> 6, ODS 0.741 vs 0.740 | `reexport_notile.py` | `results_notile.json`, `results_opset.json` |
| Per-pixel accuracy artifact (0.7726) | `diagnose.py` | printed to stdout |
| Pi latency 90.6 / 469.6 / 568.8 / 10.1 ms | `deploy/pi_benchmark.py` | `results_pi.json` |
| Figure 1 (right) | `make_figure.py` | reads `results_gate.json` |

## The gate

The contribution is four lines. `experiments/gate.py` wraps them in calibration and
evaluation, but the runtime logic is:

```python
def dark_channel_mean(img, patch=15):        # img HxWx3 float32 in [0,1]
    dc = img.min(axis=2)
    k = cv.getStructuringElement(cv.MORPH_RECT, (patch, patch))
    return float(cv.erode(dc, k).mean())

if dark_channel_mean(frame) >= TAU:          # TAU = 0.5854
    frame = dehazer(frame)
frame = edge_net(frame)
```

Under `I = Jt + A(1-t)`, the dark channel rises toward the airlight `A` as transmission
falls, so its mean is a no-reference proxy for haze density and needs no training.

`TAU` is fit once on **60 BIPEDv2 training images** by sweeping 97 candidate quantiles
and taking the one that maximises total realised F gain. The test split is never used
for calibration. The oracle in the results table does use test ground truth, by
construction, and is reported as a reference rather than an achievable policy.

## Reproducing on the Pi

See `deploy/`. Run on Python 3.7 with the bundled `tflite_runtime` wheel; newer
runtimes are unavailable for 32-bit Raspberry Pi OS (glibc 2.28). The two `.tflite`
files there are re-exported to avoid `TILE v3`, which runtime 2.11 cannot register.

```bash
python3 pi_benchmark.py --edge edge_int8.tflite --dehaze dehazer_int8.tflite \
                        --tau 0.5854 --iters 100 --threads 4
```

## superseded/

Not used for any reported number, included so the record is complete:

- `eval_edges.py`, `results_edges.json` — first edge evaluation. Downsampled the
  ground truth so thin annotated edges vanished, giving implausibly low scores.
  Replaced by `eval_edges_v2.py` (area-interpolated, thinned GT) and then by
  `eval_strict.py` (one-to-one correspondence matching).
- `baselines_dehaze.py`, `results_baselines.json` — first baseline run, on 110
  unaugmented pairs. Our own dehazer trained on 165 pairs with 8x augmentation, so
  the comparison was not like-for-like (AOD-Net 15.56 dB).
- `results_baselines_fair110.json` — second run: same augmentation, optimizer and
  schedule as ours, but still only the 110 SMOKE pairs, not the 55 DENSE-HAZE
  pairs the dehazer also saw (AOD-Net 17.08 dB).

The paper's numbers come from `results_baselines_fair.json`, the third run, on
the full 165-pair corpus. `baselines_dehaze_fair.py` trains both baselines in one
process and is what produced AOD-Net; it exhausts memory partway into DehazeNet on
an 8 GB machine, so DehazeNet was rerun by `baselines_dehazenet_165.py`, which
stages the corpus to a uint8 memmap instead of holding it in float32. Same
protocol otherwise. Raw stdout for both is in `results/logs/`. AOD-Net scores
17.08 dB on 110 and on 165 pairs alike, and DehazeNet collapses to the identity
solution on both, so the extra 55 pairs change neither baseline.
- `reexport_static.py` — an attempt to remove `TILE` by converting from a concrete
  function. Failed on unfrozen variables; `reexport_notile.py` is the approach used.

## Known limitations of this code

- `gate_macs()` in `gate.py` counts comparisons, not multiply-accumulates. It is the
  source of the "0.4% of the edge network" arithmetic figure; the measured wall-clock
  cost is 10.1 ms, which the paper reports alongside it.
- The F-measure implementation follows the standard tolerance-matching protocol but is
  our own, so absolute values are not directly comparable to published BIPED numbers.
  Use it only for comparisons under an identical protocol, as the paper does.
- `results_pipeline.json`'s SMOKE edge-consistency values (0.705 / 0.801) use the
  permissive matcher against a predicted rather than annotated target. Only their
  difference is comparable to the strict numbers elsewhere.
