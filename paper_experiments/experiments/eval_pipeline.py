"""Pipeline experiment: does dehazing before edge detection help?

A) Real SMOKE test pairs (12): dehazer PSNR/SSIM vs clean (all 8 trials);
   edge-map consistency: edge(hazy) vs edge(dehazed) compared to edge(clean).
B) Synthetic haze over BIPEDv2 test (50, real GT): ODS-F of edge(hazy),
   edge(dehazed(hazy)), edge(clean) vs ground truth, at 3 haze levels.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CNN_REPO, EDGE_REPO, SMOKE_REPO, SMOKE_DATA, RESULTS_DIR, FIGURES_DIR

import os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import cv2 as cv
from PIL import Image
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from skimage.morphology import thin
import tensorflow as tf
import eval_edges_v2 as V

EREPO = EDGE_REPO
SREPO = SMOKE_REPO
SMOKE = os.path.join(SMOKE_DATA, 'test')
OUT = {}

def load_dir(d):
    fs = sorted(os.listdir(d), key=lambda x: int(os.path.splitext(x)[0]))
    ims = []
    for f in fs:
        im = Image.open(os.path.join(d, f)).convert('RGB').resize((256,256), Image.BILINEAR)
        ims.append(np.asarray(im, dtype=np.float32)/255.0)
    return np.stack(ims), fs

hazy, hf = load_dir(f'{SMOKE}/hazy')
clean, cf = load_dir(f'{SMOKE}/clean')
print('SMOKE test:', hazy.shape, clean.shape, hf[:3], cf[:3], flush=True)

# ---- A1: dehazer PSNR/SSIM on real smoke, all trials ----
per_trial = {}
best_trial, best_psnr, best_out = None, -1, None
for t in range(1, 9):
    try:
        m = tf.keras.models.load_model(f'{SREPO}/saved_models/dehazer_trial_{t}.keras', compile=False)
    except Exception as e:
        # trial 6 contains a Lambda layer with marshaled bytecode that will not
        # deserialize under this Python version; skip it rather than guess.
        print(f'dehazer trial {t}: SKIPPED ({str(e)[:60]}...)', flush=True)
        per_trial[t] = 'unloadable'
        continue
    out = m.predict(hazy, batch_size=4, verbose=0)
    ps = float(np.mean([psnr(clean[i], out[i], data_range=1.0) for i in range(len(clean))]))
    ss = float(np.mean([ssim(clean[i], out[i], data_range=1.0, channel_axis=2) for i in range(len(clean))]))
    per_trial[t] = {'PSNR': round(ps,2), 'SSIM': round(ss,3)}
    print(f'dehazer trial {t}: PSNR={ps:.2f} SSIM={ss:.3f}', flush=True)
    if ps > best_psnr: best_trial, best_psnr, best_out = t, ps, out
# input as baseline (no dehazing)
ps0 = float(np.mean([psnr(clean[i], hazy[i], data_range=1.0) for i in range(len(clean))]))
ss0 = float(np.mean([ssim(clean[i], hazy[i], data_range=1.0, channel_axis=2) for i in range(len(clean))]))
OUT['dehazer_smoke_test'] = {'per_trial': per_trial, 'best_trial': best_trial,
                             'input_baseline': {'PSNR': round(ps0,2), 'SSIM': round(ss0,3)}}
print(f'no-dehaze input baseline: PSNR={ps0:.2f} SSIM={ss0:.3f} | best trial {best_trial}', flush=True)

# ---- A2: edge consistency on real smoke ----
def edge_consistency(edge_model, dehazed):
    e_clean = edge_model.predict(clean, batch_size=4, verbose=0)[...,0]
    e_hazy  = edge_model.predict(hazy, batch_size=4, verbose=0)[...,0]
    e_dhz   = edge_model.predict(dehazed, batch_size=4, verbose=0)[...,0]
    res = {}
    for name, e in [('hazy', e_hazy), ('dehazed', e_dhz)]:
        ss = float(np.mean([ssim(e_clean[i], e[i], data_range=1.0) for i in range(len(e))]))
        # tolerance-F vs binarized edge(clean) as pseudo-GT
        s = [0,0,0,0]
        for i in range(len(e)):
            gt = thin(e_clean[i] > 0.1)
            c = V.match_counts(e[i] > 0.1, gt)
            s = [a+b for a,b in zip(s,c)]
        f = V.f_from_counts(*s)[0]
        res[name] = {'SSIM_vs_edge_clean': round(ss,3), 'F_vs_edge_clean': round(f,3)}
    return res

for et in [7, 13]:
    em = tf.keras.models.load_model(f'{EREPO}/saved_models/cnn_trial_{et}.keras', compile=False)
    r = edge_consistency(em, best_out)
    OUT[f'smoke_edge_consistency_trial{et}'] = r
    print(f'edge trial {et} consistency:', r, flush=True)

# ---- B: synthetic haze over BIPEDv2 test with real GT ----
imgs, gt_bins, gt_softs, files = V.load_pairs()
dm = tf.keras.models.load_model(f'{SREPO}/saved_models/dehazer_trial_{best_trial}.keras', compile=False)

def ods_f(preds, gt_bins):
    best = 0
    for t in V.THRESHOLDS:
        s = [0,0,0,0]
        for i in range(len(preds)):
            c = V.match_counts(preds[i] > t, gt_bins[i])
            s = [a+b for a,b in zip(s,c)]
        best = max(best, V.f_from_counts(*s)[0])
    return round(best,3)

A = 0.8
for et in [7, 13]:
    em = tf.keras.models.load_model(f'{EREPO}/saved_models/cnn_trial_{et}.keras', compile=False)
    e_clean = em.predict(imgs, batch_size=8, verbose=0)[...,0]
    row = {'clean': ods_f(list(e_clean), gt_bins)}
    for tmap in [0.7, 0.5, 0.35]:
        hz = np.clip(imgs*tmap + A*(1-tmap), 0, 1).astype(np.float32)
        dh = dm.predict(hz, batch_size=4, verbose=0)
        e_h = em.predict(hz, batch_size=8, verbose=0)[...,0]
        e_d = em.predict(dh, batch_size=8, verbose=0)[...,0]
        row[f't={tmap}'] = {'hazy': ods_f(list(e_h), gt_bins), 'dehazed': ods_f(list(e_d), gt_bins)}
        print(f'edge trial {et} t={tmap}:', row[f't={tmap}'], flush=True)
    OUT[f'synthetic_biped_trial{et}'] = row

with open(os.path.join(RESULTS_DIR, 'results_pipeline.json'),'w') as f:
    json.dump(OUT, f, indent=2)
print(json.dumps(OUT, indent=2))
