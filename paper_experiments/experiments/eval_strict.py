"""Strict BSDS-style edge evaluation with one-to-one correspondence matching.

The earlier protocol counted every predicted pixel inside the dilated GT, so one
GT pixel could absolve many predictions (permissive). Here each predicted pixel
may be matched to at most one GT pixel and vice versa, solved as a bipartite
assignment on the sparse within-tolerance pairs (scipy linear_sum_assignment on
a cost matrix restricted to candidates, per connected chunk to stay tractable).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CNN_REPO, EDGE_REPO, SMOKE_REPO, SMOKE_DATA, RESULTS_DIR, FIGURES_DIR

import os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import cv2 as cv
from skimage.morphology import thin
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import tensorflow as tf
import eval_edges_v2 as V

REPO = V.REPO
THRESHOLDS = np.arange(0.05, 1.0, 0.05)


def match_one_to_one(pred_bin, gt_bin, r=2):
    """Return (matched, n_pred, n_gt) under a one-to-one correspondence within r px."""
    pred_bin = thin(pred_bin)
    P = np.argwhere(pred_bin)
    G = np.argwhere(gt_bin)
    if len(P) == 0 or len(G) == 0:
        return 0, len(P), len(G)

    # candidate pairs within tolerance
    tree = cKDTree(G)
    pairs = tree.query_ball_point(P, r=r)
    cand_p, cand_g = [], []
    for i, gl in enumerate(pairs):
        for j in gl:
            cand_p.append(i); cand_g.append(j)
    if not cand_p:
        return 0, len(P), len(G)
    cand_p = np.asarray(cand_p); cand_g = np.asarray(cand_g)

    # restrict to the involved nodes, then solve assignment on that subgraph.
    up, ip = np.unique(cand_p, return_inverse=True)
    ug, ig = np.unique(cand_g, return_inverse=True)
    n, m = len(up), len(ug)

    # dense cost on the reduced problem; BIG for non-candidate pairs
    BIG = 1e6
    cost = np.full((n, m), BIG, dtype=np.float64)
    d = np.linalg.norm(P[cand_p] - G[cand_g], axis=1)
    cost[ip, ig] = np.minimum(cost[ip, ig], d)

    if n * m > 40_000_000:   # guard: fall back to greedy for pathological sizes
        order = np.argsort(d)
        usedp, usedg, matched = set(), set(), 0
        for k in order:
            a, b = cand_p[k], cand_g[k]
            if a not in usedp and b not in usedg:
                usedp.add(a); usedg.add(b); matched += 1
        return matched, len(P), len(G)

    ri, ci = linear_sum_assignment(cost)
    matched = int((cost[ri, ci] < BIG).sum())
    return matched, len(P), len(G)


def prf(matched, n_pred, n_gt):
    prec = matched / n_pred if n_pred else 0.0
    rec = matched / n_gt if n_gt else 0.0
    f = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f, prec, rec


def ods_ois_strict(preds, gt_bins, r=2):
    n = len(preds)
    counts = {t: [] for t in THRESHOLDS}
    for i in range(n):
        for t in THRESHOLDS:
            counts[t].append(match_one_to_one(preds[i] > t, gt_bins[i], r))
    def agg(t):
        c = counts[t]
        return prf(sum(x[0] for x in c), sum(x[1] for x in c), sum(x[2] for x in c))[0]
    ods_t, ods = max(((t, agg(t)) for t in THRESHOLDS), key=lambda x: x[1])
    best = [max(THRESHOLDS, key=lambda t: prf(*counts[t][i])[0]) for i in range(n)]
    ois = prf(sum(counts[best[i]][i][0] for i in range(n)),
              sum(counts[best[i]][i][1] for i in range(n)),
              sum(counts[best[i]][i][2] for i in range(n)))[0]
    return round(ods, 3), round(float(ods_t), 2), round(ois, 3)


def f_strict_binary(preds_bin, gt_bins, r=2):
    c = [match_one_to_one(p, g, r) for p, g in zip(preds_bin, gt_bins)]
    return round(prf(sum(x[0] for x in c), sum(x[1] for x in c), sum(x[2] for x in c))[0], 3)


def main():
    imgs, gt_bins, gt_softs, files = V.load_pairs()
    OUT = {}

    for trial in [7, 13]:
        m = tf.keras.models.load_model(f'{REPO}/saved_models/cnn_trial_{trial}.keras', compile=False)
        preds = list(m.predict(imgs, batch_size=8, verbose=0)[..., 0])
        ods, t, ois = ods_ois_strict(preds, gt_bins)
        OUT[f'cnn_trial_{trial}'] = {'ODS': ods, 'ODS_thresh': t, 'OIS': ois}
        print(f'trial {trial} strict:', OUT[f'cnn_trial_{trial}'], flush=True)

    best = None
    for high in [40, 60, 80, 100, 150]:
        pb = []
        for im in imgs:
            g = cv.cvtColor((im * 255).astype(np.uint8), cv.COLOR_RGB2GRAY)
            g = cv.GaussianBlur(g, (5, 5), 1.4)
            pb.append(cv.Canny(g, high // 2, high) > 0)
        f = f_strict_binary(pb, gt_bins)
        print('canny strict', high, f, flush=True)
        if best is None or f > best[1]:
            best = (high, f)
    OUT['canny_strict'] = {'F': best[1], 'high': best[0]}

    # quantized model under the same strict protocol
    interp = tf.lite.Interpreter(model_path=f'{REPO}/lite/lite-models/model_10.tflite')
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]; out = interp.get_output_details()[0]
    qpreds = []
    for im in imgs:
        x = np.expand_dims((im * 255).astype(np.uint8), 0)
        interp.set_tensor(inp['index'], x); interp.invoke()
        y = interp.get_tensor(out['index'])[0]
        sc, zp = out['quantization']
        y = (y.astype(np.float32) - zp) * sc if sc else y.astype(np.float32) / 255.0
        qpreds.append(np.squeeze(y))
    ods, t, ois = ods_ois_strict(qpreds, gt_bins)
    OUT['cnn_trial_13_uint8'] = {'ODS': ods, 'ODS_thresh': t, 'OIS': ois}
    print('uint8 strict:', OUT['cnn_trial_13_uint8'], flush=True)

    json.dump(OUT, open(os.path.join(RESULTS_DIR, 'results_strict.json'), 'w'), indent=2)
    print(json.dumps(OUT, indent=2))


if __name__ == '__main__':
    main()
