"""Proper edge-detection evaluation on BIPEDv2 test set.

Computes tolerance-matched ODS/OIS F-measure + SSIM for CNN trials and Canny,
plus the accuracy diagnosis (all-background baseline).
"""
import os, json, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import cv2 as cv
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf

REPO = '/home/claude/dual-cnn-system-firefighters/edge'
IMG_DIR = f'{REPO}/BIPEDv2/BIPED/edges/imgs/test/rgbr/'
GT_DIR  = f'{REPO}/BIPEDv2/BIPED/edges/edge_maps/test/rgbr/'
SIZE = (256, 256)
TOL = 2  # matching tolerance in px at 256x256 (~0.0075 * diagonal)
THRESHOLDS = np.arange(0.05, 1.0, 0.05)

def load_pairs():
    imgs, gts = [], []
    files = sorted(os.listdir(IMG_DIR))
    for f in files:
        im = Image.open(os.path.join(IMG_DIR, f)).convert('RGB').resize(SIZE)
        imgs.append(np.asarray(im, dtype=np.float32) / 255.0)
        g = Image.open(os.path.join(GT_DIR, f.replace('.jpg', '.png'))).convert('L').resize(SIZE)
        gts.append(np.asarray(g, dtype=np.float32) / 255.0)
    return np.stack(imgs), np.stack(gts), files

def dilate(mask, r):
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    return cv.dilate(mask.astype(np.uint8), k).astype(bool)

def match_counts(pred_bin, gt_bin, tol=TOL):
    """Tolerance-matched counts: TP_p (pred pixels near GT), P, TP_r (GT pixels near pred), G."""
    gt_d = dilate(gt_bin, tol)
    pr_d = dilate(pred_bin, tol)
    P = int(pred_bin.sum()); G = int(gt_bin.sum())
    TPp = int((pred_bin & gt_d).sum())
    TPr = int((gt_bin & pr_d).sum())
    return TPp, P, TPr, G

def f_from_counts(TPp, P, TPr, G):
    prec = TPp / P if P else 0.0
    rec = TPr / G if G else 0.0
    return 2*prec*rec/(prec+rec) if prec+rec else 0.0, prec, rec

def eval_soft_predictions(preds, gts):
    """ODS/OIS over threshold sweep for soft (0-1) predictions."""
    gt_bins = [g > 0.5 for g in gts]
    n = len(preds)
    # per-threshold aggregate counts and per-image F
    agg = {t: [0,0,0,0] for t in THRESHOLDS}
    per_img_best = np.zeros(n)
    for i in range(n):
        best = 0.0
        for t in THRESHOLDS:
            c = match_counts(preds[i] > t, gt_bins[i])
            for j in range(4): agg[t][j] += c[j]
            f,_,_ = f_from_counts(*c)
            best = max(best, f)
        per_img_best[i] = best
    ods_t, ods = max(((t, f_from_counts(*agg[t])[0]) for t in THRESHOLDS), key=lambda x: x[1])
    ois = float(per_img_best.mean())
    mssim = float(np.mean([ssim(preds[i], gts[i], data_range=1.0) for i in range(n)]))
    return {'ODS': round(ods,4), 'ODS_thresh': round(float(ods_t),2), 'OIS': round(ois,4), 'SSIM': round(mssim,4)}

def eval_binary_predictions(preds_bin, gts):
    gt_bins = [g > 0.5 for g in gts]
    agg = [0,0,0,0]; per_img = []
    for i in range(len(preds_bin)):
        c = match_counts(preds_bin[i], gt_bins[i])
        for j in range(4): agg[j] += c[j]
        per_img.append(f_from_counts(*c)[0])
    f,_,_ = f_from_counts(*agg)
    mssim = float(np.mean([ssim(preds_bin[i].astype(np.float32), gts[i], data_range=1.0) for i in range(len(preds_bin))]))
    return {'F': round(f,4), 'F_per_img_mean': round(float(np.mean(per_img)),4), 'SSIM': round(mssim,4)}

def main():
    imgs, gts, files = load_pairs()
    print(f'{len(imgs)} test pairs loaded', flush=True)
    results = {}

    # accuracy diagnosis
    gt_bins = np.stack([g > 0.5 for g in gts])
    edge_frac = gt_bins.mean()
    results['diagnosis'] = {
        'edge_pixel_fraction': round(float(edge_frac),4),
        'all_background_accuracy': round(float(1-edge_frac),4),
    }

    # CNN trials
    for trial in [7, 11, 13]:
        path = f'{REPO}/saved_models/cnn_trial_{trial}.keras'
        m = tf.keras.models.load_model(path, compile=False)
        preds = m.predict(imgs, batch_size=8, verbose=0)[..., 0]
        r = eval_soft_predictions(list(preds), list(gts))
        # accuracy at 0.5 for diagnosis
        acc = float(np.mean((preds > 0.5) == gt_bins))
        r['pixel_accuracy_at_0.5'] = round(acc,4)
        results[f'cnn_trial_{trial}'] = r
        print(f'trial {trial}:', r, flush=True)

    # Canny sweep (low = high/2), on 8-bit grayscale
    best = None
    for high in [60, 100, 150, 200, 250]:
        preds_bin = []
        for im in imgs:
            gray = cv.cvtColor((im*255).astype(np.uint8), cv.COLOR_RGB2GRAY)
            gray = cv.GaussianBlur(gray, (5,5), 1.4)
            e = cv.Canny(gray, high//2, high) > 0
            preds_bin.append(e)
        r = eval_binary_predictions(preds_bin, list(gts))
        r['high_thresh'] = high
        print('canny', high, r, flush=True)
        if best is None or r['F'] > best['F']: best = r
    results['canny_best'] = best

    with open('/home/claude/experiments/results_edges.json','w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
