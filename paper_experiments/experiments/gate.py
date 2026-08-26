"""A realized haze-density gate, calibrated on train and evaluated on test.

Estimator: mean of the dark channel of the input. Under the atmospheric model
I = Jt + A(1-t), as t falls the dark channel rises toward A, so the dark-channel
mean is a monotone no-reference proxy for haze density. Cost is a 3-channel min
plus a 15x15 min-filter: ~0.001 GMAC at 256x256, i.e. 0.2% of the edge network
and 0.04% of the dehazer, so the gate is free relative to what it saves.

Protocol:
  - threshold tau chosen on BIPEDv2 TRAIN images (never the test split)
  - evaluated on the 50 test images at 4 conditions (clean + 3 haze levels)
  - compared against always-dehaze, never-dehaze, and an oracle upper bound
Reports ODS under the strict one-to-one protocol and mean MAC/frame.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CNN_REPO, EDGE_REPO, SMOKE_REPO, SMOKE_DATA, RESULTS_DIR, FIGURES_DIR

import os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import cv2 as cv
from PIL import Image
from skimage.morphology import thin
import tensorflow as tf
import eval_edges_v2 as V
from eval_strict import ods_ois_strict, match_one_to_one, prf, THRESHOLDS

REPO = V.REPO
SREPO = SMOKE_REPO
MAC_EDGE = 523_436_032
MAC_DEHAZE = 2_283_798_528
A_LIGHT = 0.8
LEVELS = [('clean', None), ('t=0.7', 0.7), ('t=0.5', 0.5), ('t=0.35', 0.35)]


# ---------------- the gate ----------------
def dark_channel_mean(img, patch=15):
    """img float32 HxWx3 in [0,1] -> scalar haze score in [0,1]."""
    dc = img.min(axis=2)
    k = cv.getStructuringElement(cv.MORPH_RECT, (patch, patch))
    return float(cv.erode(dc, k).mean())


def gate_macs(h, w, patch=15):
    """Comparisons per frame for min over 3 channels + separable min-filter."""
    return h * w * (2 + 2 * (patch - 1))


# ---------------- data ----------------
def haze(imgs, t):
    if t is None:
        return imgs
    return np.clip(imgs * t + A_LIGHT * (1 - t), 0, 1).astype(np.float32)


def load_split(img_dir, gt_dir, limit=None):
    imgs, gts = [], []
    files = sorted(os.listdir(img_dir))
    if limit:
        files = files[:limit]
    for f in files:
        im = Image.open(os.path.join(img_dir, f)).convert('RGB').resize((256, 256), Image.BILINEAR)
        imgs.append(np.asarray(im, dtype=np.float32) / 255.0)
        gpath = os.path.join(gt_dir, f.replace('.jpg', '.png'))
        g = cv.imread(gpath, cv.IMREAD_GRAYSCALE)
        gb = (g > 127).astype(np.float32)
        gs = cv.resize(gb, (256, 256), interpolation=cv.INTER_AREA)
        gts.append(thin(gs > 0.15))
    return np.stack(imgs), gts


BASE = f'{REPO}/BIPEDv2/BIPED/edges'
train_imgs, train_gts = load_split(f'{BASE}/imgs/train/rgbr/real/', f'{BASE}/edge_maps/train/rgbr/real/', limit=60)
test_imgs, test_gts = load_split(f'{BASE}/imgs/test/rgbr/', f'{BASE}/edge_maps/test/rgbr/')
print(f'calibration set {len(train_imgs)}, test set {len(test_imgs)}', flush=True)

edge = tf.keras.models.load_model(f'{REPO}/saved_models/cnn_trial_13.keras', compile=False)
try:
    dehaze = tf.keras.models.load_model(f'{SREPO}/saved_models/dehazer_trial_8.keras', compile=False)
except Exception:
    dehaze = tf.keras.models.load_model(f'{SREPO}/saved_models/dehazer_trial_8.keras', compile=False, safe_mode=False)


def ods_of(preds, gts):
    return ods_ois_strict(preds, gts)[0]


def per_image_f(pred, gt):
    """best-threshold F for a single image (used for the oracle)."""
    return max(prf(*match_one_to_one(pred > t, gt))[0] for t in THRESHOLDS)


# ---------------- 1) calibrate tau on TRAIN ----------------
cal = []   # (haze_score, gain_from_dehazing)
for name, t in LEVELS:
    x = haze(train_imgs, t)
    e_h = edge.predict(x, batch_size=8, verbose=0)[..., 0]
    d = dehaze.predict(x, batch_size=4, verbose=0)
    e_d = edge.predict(d, batch_size=8, verbose=0)[..., 0]
    for i in range(len(x)):
        s = dark_channel_mean(x[i])
        gain = per_image_f(e_d[i], train_gts[i]) - per_image_f(e_h[i], train_gts[i])
        cal.append((s, gain))
    print(f'calibrated on {name}', flush=True)

cal = np.array(cal)
scores, gains = cal[:, 0], cal[:, 1]
# choose tau maximising total realised gain: dehaze iff score >= tau
cands = np.quantile(scores, np.linspace(0.02, 0.98, 97))
tot = [(tau, gains[scores >= tau].sum()) for tau in cands]
TAU = float(max(tot, key=lambda z: z[1])[0])
print(f'calibrated tau = {TAU:.4f}', flush=True)

# ---------------- 2) evaluate on TEST ----------------
rows = {}
gate_frac = {}
for name, t in LEVELS:
    x = haze(test_imgs, t)
    e_h = list(edge.predict(x, batch_size=8, verbose=0)[..., 0])
    d = dehaze.predict(x, batch_size=4, verbose=0)
    e_d = list(edge.predict(d, batch_size=8, verbose=0)[..., 0])
    s = np.array([dark_channel_mean(im) for im in x])
    use = s >= TAU
    gated = [e_d[i] if use[i] else e_h[i] for i in range(len(x))]
    # oracle: per-image best of the two
    oracle = []
    for i in range(len(x)):
        oracle.append(e_d[i] if per_image_f(e_d[i], test_gts[i]) >= per_image_f(e_h[i], test_gts[i]) else e_h[i])
    rows[name] = {
        'never_dehaze': ods_of(e_h, test_gts),
        'always_dehaze': ods_of(e_d, test_gts),
        'gated': ods_of(gated, test_gts),
        'oracle': ods_of(oracle, test_gts),
        'dehaze_rate': round(float(use.mean()), 3),
        'mean_haze_score': round(float(s.mean()), 4),
    }
    gate_frac[name] = float(use.mean())
    print(name, rows[name], flush=True)

# ---------------- 3) cost ----------------
g_macs = gate_macs(256, 256)
cost = {}
for name in rows:
    fr = gate_frac[name]
    cost[name] = {
        'always_GMAC': round((MAC_EDGE + MAC_DEHAZE) / 1e9, 2),
        'gated_GMAC': round((MAC_EDGE + fr * MAC_DEHAZE + g_macs) / 1e9, 2),
        'never_GMAC': round((MAC_EDGE + g_macs) / 1e9, 2),
    }
mean_fr = float(np.mean(list(gate_frac.values())))
OUT = {
    'tau': round(TAU, 4),
    'gate_cost_ops_per_frame': int(g_macs),
    'gate_cost_pct_of_edge': round(100 * g_macs / MAC_EDGE, 3),
    'per_level': rows,
    'cost_GMAC': cost,
    'mean_dehaze_rate': round(mean_fr, 3),
    'mean_gated_GMAC': round((MAC_EDGE + mean_fr * MAC_DEHAZE + g_macs) / 1e9, 2),
    'always_GMAC': round((MAC_EDGE + MAC_DEHAZE) / 1e9, 2),
}
# mean ODS across the 4 conditions
for k in ['never_dehaze', 'always_dehaze', 'gated', 'oracle']:
    OUT[f'mean_ODS_{k}'] = round(float(np.mean([rows[n][k] for n in rows])), 3)

json.dump(OUT, open(os.path.join(RESULTS_DIR, 'results_gate.json'), 'w'), indent=2)
print(json.dumps(OUT, indent=2))
