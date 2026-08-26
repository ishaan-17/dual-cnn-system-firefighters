"""Edge evaluation v2: fixes GT downsampling (area-interp + low threshold so thin
edges survive), thins binarized predictions before tolerance matching (closer to
standard BSDS practice), and computes OIS by aggregating counts at per-image best
thresholds."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CNN_REPO, EDGE_REPO, SMOKE_REPO, SMOKE_DATA, RESULTS_DIR, FIGURES_DIR

import os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import cv2 as cv
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import thin
import tensorflow as tf

REPO = EDGE_REPO
IMG_DIR = f'{REPO}/BIPEDv2/BIPED/edges/imgs/test/rgbr/'
GT_DIR  = f'{REPO}/BIPEDv2/BIPED/edges/edge_maps/test/rgbr/'
SIZE = (256, 256)
TOL = 2
THRESHOLDS = np.arange(0.05, 1.0, 0.05)

def load_pairs():
    imgs, gt_bins, gt_softs = [], [], []
    files = sorted(os.listdir(IMG_DIR))
    for f in files:
        im = Image.open(os.path.join(IMG_DIR, f)).convert('RGB').resize(SIZE, Image.BILINEAR)
        imgs.append(np.asarray(im, dtype=np.float32) / 255.0)
        g = cv.imread(os.path.join(GT_DIR, f.replace('.jpg', '.png')), cv.IMREAD_GRAYSCALE)
        gb = (g > 127).astype(np.float32)
        gs = cv.resize(gb, SIZE, interpolation=cv.INTER_AREA)  # coverage fraction
        gt_softs.append(gs)
        gt_bins.append(thin(gs > 0.15))  # binarize + thin to ~1px GT
    return np.stack(imgs), gt_bins, gt_softs, files

def dilate(mask, r=TOL):
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    return cv.dilate(mask.astype(np.uint8), k).astype(bool)

def match_counts(pred_bin, gt_bin):
    pred_bin = thin(pred_bin)  # thin prediction to ~1px before matching
    gt_d = dilate(gt_bin); pr_d = dilate(pred_bin)
    return (int((pred_bin & gt_d).sum()), int(pred_bin.sum()),
            int((gt_bin & pr_d).sum()), int(gt_bin.sum()))

def f_from_counts(TPp, P, TPr, G):
    prec = TPp / P if P else 0.0
    rec = TPr / G if G else 0.0
    return (2*prec*rec/(prec+rec) if prec+rec else 0.0), prec, rec

def eval_soft(preds, gt_bins, gt_softs):
    n = len(preds)
    counts = {t: [] for t in THRESHOLDS}   # per image per threshold
    for i in range(n):
        for t in THRESHOLDS:
            counts[t].append(match_counts(preds[i] > t, gt_bins[i]))
    # ODS: best fixed threshold on aggregated counts
    def agg_f(t, idxs=None):
        cs = counts[t] if idxs is None else [counts[t][i] for i in idxs]
        s = [sum(c[j] for c in cs) for j in range(4)]
        return f_from_counts(*s)[0]
    ods_t, ods = max(((t, agg_f(t)) for t in THRESHOLDS), key=lambda x: x[1])
    # OIS: per-image best threshold, aggregate those counts
    best_ts = []
    for i in range(n):
        bt = max(THRESHOLDS, key=lambda t: f_from_counts(*counts[t][i])[0])
        best_ts.append(bt)
    s = [sum(counts[best_ts[i]][i][j] for i in range(n)) for j in range(4)]
    ois = f_from_counts(*s)[0]
    mssim = float(np.mean([ssim(preds[i], gt_softs[i], data_range=1.0) for i in range(n)]))
    return {'ODS': round(ods,3), 'ODS_thresh': round(float(ods_t),2),
            'OIS': round(ois,3), 'SSIM': round(mssim,3)}

def eval_binary(preds_bin, gt_bins, gt_softs):
    n = len(preds_bin)
    cs = [match_counts(preds_bin[i], gt_bins[i]) for i in range(n)]
    s = [sum(c[j] for c in cs) for j in range(4)]
    f, prec, rec = f_from_counts(*s)
    mssim = float(np.mean([ssim(preds_bin[i].astype(np.float32), gt_softs[i], data_range=1.0) for i in range(n)]))
    return {'F': round(f,3), 'precision': round(prec,3), 'recall': round(rec,3), 'SSIM': round(mssim,3)}

def main():
    imgs, gt_bins, gt_softs, files = load_pairs()
    frac = float(np.mean([g.mean() for g in [gs > 0.15 for gs in gt_softs]]))
    results = {'gt_edge_fraction_unthinned': round(frac,4)}
    print(f'{len(imgs)} pairs, GT edge fraction (unthinned): {frac:.4f}', flush=True)

    for trial in [7, 11, 13]:
        m = tf.keras.models.load_model(f'{REPO}/saved_models/cnn_trial_{trial}.keras', compile=False)
        preds = m.predict(imgs, batch_size=8, verbose=0)[..., 0]
        r = eval_soft(list(preds), gt_bins, gt_softs)
        results[f'cnn_trial_{trial}'] = r
        print(f'trial {trial}:', r, flush=True)

    best = None
    for high in [60, 100, 150, 200, 250]:
        preds_bin = []
        for im in imgs:
            gray = cv.cvtColor((im*255).astype(np.uint8), cv.COLOR_RGB2GRAY)
            gray = cv.GaussianBlur(gray, (5,5), 1.4)
            preds_bin.append(cv.Canny(gray, high//2, high) > 0)
        r = eval_binary(preds_bin, gt_bins, gt_softs); r['high_thresh'] = high
        print('canny', high, r, flush=True)
        if best is None or r['F'] > best['F']: best = r
    results['canny_best'] = best

    with open(os.path.join(RESULTS_DIR, 'results_edges_v2.json'),'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
