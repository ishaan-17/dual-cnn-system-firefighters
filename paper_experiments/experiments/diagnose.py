"""1) Robustness: rerun trial 7 + best Canny with tolerance r=1.
2) Reproduce the constant 0.7726 test accuracy using the repo's own loading code."""
import os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import cv2 as cv
from PIL import Image
from skimage.morphology import thin
import tensorflow as tf
import eval_edges_v2 as V

# ---- 1) tolerance robustness ----
imgs, gt_bins, gt_softs, files = V.load_pairs()

def match_counts_r(pred_bin, gt_bin, r):
    pred_bin = thin(pred_bin)
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    gt_d = cv.dilate(gt_bin.astype(np.uint8), k).astype(bool)
    pr_d = cv.dilate(pred_bin.astype(np.uint8), k).astype(bool)
    return (int((pred_bin & gt_d).sum()), int(pred_bin.sum()),
            int((gt_bin & pr_d).sum()), int(gt_bin.sum()))

m = tf.keras.models.load_model(f'{V.REPO}/saved_models/cnn_trial_7.keras', compile=False)
preds = m.predict(imgs, batch_size=8, verbose=0)[..., 0]

for r in [1, 2]:
    # CNN at threshold sweep -> ODS
    best_f = 0
    for t in V.THRESHOLDS:
        s = [0,0,0,0]
        for i in range(len(imgs)):
            c = match_counts_r(preds[i] > t, gt_bins[i], r)
            s = [a+b for a,b in zip(s,c)]
        f = V.f_from_counts(*s)[0]
        best_f = max(best_f, f)
    # canny high=60
    s = [0,0,0,0]
    for i in range(len(imgs)):
        gray = cv.cvtColor((imgs[i]*255).astype(np.uint8), cv.COLOR_RGB2GRAY)
        gray = cv.GaussianBlur(gray, (5,5), 1.4)
        c = match_counts_r(cv.Canny(gray, 30, 60) > 0, gt_bins[i], r)
        s = [a+b for a,b in zip(s,c)]
    fc = V.f_from_counts(*s)[0]
    print(f'tolerance r={r}: CNN(t7) ODS={best_f:.3f}  Canny(60) F={fc:.3f}', flush=True)

# ---- 2) reproduce their test accuracy ----
def their_load(directory, image_size=(256,256)):
    fs = sorted([f for f in os.listdir(directory) if f.endswith(('.png','.jpg','.jpeg'))])
    arr = np.empty((len(fs), 256, 256, 1 if 'edge_maps' in directory else 3), dtype='float32')
    for i, fn in enumerate(fs):
        image = Image.open(os.path.join(directory, fn))
        if 'edge_maps' in directory:
            image.convert('L')
        image = image.resize(image_size)  # PIL default resample (same as their code)
        if 'edge_maps' in directory:
            arr[i,:,:,0] = np.array(image)/255.0
        else:
            arr[i] = np.array(image)/255.0
    return arr

ti = their_load(f'{V.REPO}/BIPEDv2/BIPED/edges/imgs/test/rgbr/')
tl = their_load(f'{V.REPO}/BIPEDv2/BIPED/edges/edge_maps/test/rgbr/')
print('their GT>0.5 fraction:', round(float((tl>0.5).mean()),4),
      '| GT nonzero fraction:', round(float((tl>0).mean()),4), flush=True)
for trial in [7, 13]:
    m = tf.keras.models.load_model(f'{V.REPO}/saved_models/cnn_trial_{trial}.keras', compile=False)
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    loss, acc = m.evaluate(ti, tl, verbose=0)
    p = m.predict(ti, batch_size=8, verbose=0)
    print(f'trial {trial}: evaluate acc={acc:.6f} | frac pred>0.5={float((p>0.5).mean()):.4f} '
          f'| pred min/mean/max={p.min():.3f}/{p.mean():.3f}/{p.max():.3f}', flush=True)
