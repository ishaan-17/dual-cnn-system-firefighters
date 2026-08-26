"""Fill the gaps the fact-check found:
 - trial 13 (the deployed 23K model) at r=1, with Canny RE-SWEPT at r=1 too
 - parameter counts for every edge trial, so the paper attributes them correctly
 - trial 13 gap-recovery fraction at t=0.35
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CNN_REPO, EDGE_REPO, SMOKE_REPO, SMOKE_DATA, RESULTS_DIR, FIGURES_DIR

import os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np, cv2 as cv
from skimage.morphology import thin
import tensorflow as tf
import eval_edges_v2 as V

imgs, gt_bins, gt_softs, files = V.load_pairs()
OUT = {}

# --- parameter counts ---
params = {}
for t in [7, 11, 13]:
    m = tf.keras.models.load_model(f'{V.REPO}/saved_models/cnn_trial_{t}.keras', compile=False)
    params[t] = int(m.count_params())
OUT['edge_params'] = params
print('params:', params, flush=True)

def match_r(pred_bin, gt_bin, r):
    pred_bin = thin(pred_bin)
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    gt_d = cv.dilate(gt_bin.astype(np.uint8), k).astype(bool)
    pr_d = cv.dilate(pred_bin.astype(np.uint8), k).astype(bool)
    return (int((pred_bin & gt_d).sum()), int(pred_bin.sum()),
            int((gt_bin & pr_d).sum()), int(gt_bin.sum()))

def ods(preds, r):
    best = 0
    for t in V.THRESHOLDS:
        s = [0,0,0,0]
        for i in range(len(preds)):
            c = match_r(preds[i] > t, gt_bins[i], r)
            s = [a+b for a,b in zip(s,c)]
        best = max(best, V.f_from_counts(*s)[0])
    return round(best,3)

def canny_best(r):
    best, bh = 0, None
    for high in [40, 60, 80, 100, 150, 200, 250]:
        s = [0,0,0,0]
        for im in imgs:
            g = cv.cvtColor((im*255).astype(np.uint8), cv.COLOR_RGB2GRAY)
            g = cv.GaussianBlur(g, (5,5), 1.4)
            c = match_r(cv.Canny(g, high//2, high) > 0, gt_bins[imgs.tolist().index(im.tolist())] if False else gt_bins[0], r)
        # (recomputed properly below)
        break
    return None

# proper Canny sweep at a given tolerance
def canny_sweep(r):
    results = {}
    for high in [40, 60, 80, 100, 150, 200, 250]:
        s = [0,0,0,0]
        for i, im in enumerate(imgs):
            g = cv.cvtColor((im*255).astype(np.uint8), cv.COLOR_RGB2GRAY)
            g = cv.GaussianBlur(g, (5,5), 1.4)
            c = match_r(cv.Canny(g, high//2, high) > 0, gt_bins[i], r)
            s = [a+b for a,b in zip(s,c)]
        results[high] = round(V.f_from_counts(*s)[0], 3)
    bh = max(results, key=results.get)
    return {'best_F': results[bh], 'best_high': bh, 'sweep': results}

for r in [1, 2]:
    m13 = tf.keras.models.load_model(f'{V.REPO}/saved_models/cnn_trial_13.keras', compile=False)
    p13 = list(m13.predict(imgs, batch_size=8, verbose=0)[...,0])
    m7 = tf.keras.models.load_model(f'{V.REPO}/saved_models/cnn_trial_7.keras', compile=False)
    p7 = list(m7.predict(imgs, batch_size=8, verbose=0)[...,0])
    cs = canny_sweep(r)
    OUT[f'r{r}'] = {'trial13_ODS': ods(p13, r), 'trial7_ODS': ods(p7, r), 'canny': cs}
    print(f'r={r}:', OUT[f'r{r}'], flush=True)

# gap recovery at t=0.35 for both models
pipe = json.load(open(os.path.join(RESULTS_DIR, 'results_pipeline.json')))
for tr in [7, 13]:
    d = pipe[f'synthetic_biped_trial{tr}']
    clean, hazy, deh = d['clean'], d['t=0.35']['hazy'], d['t=0.35']['dehazed']
    OUT[f'gap_recovery_trial{tr}_t035'] = round((deh-hazy)/(clean-hazy), 3)
    print(f'trial {tr} gap recovery at t=0.35:', OUT[f'gap_recovery_trial{tr}_t035'], flush=True)

json.dump(OUT, open(os.path.join(RESULTS_DIR, 'results_fix.json'),'w'), indent=2)
print(json.dumps(OUT, indent=2))
