"""Two cheap but high-value additions:
 1) Quantization ablation: float32 Keras edge model vs uint8 TFLite, same protocol.
 2) Dark channel prior baseline for dehazing on the real SMOKE test set.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CNN_REPO, EDGE_REPO, SMOKE_REPO, SMOKE_DATA, RESULTS_DIR, FIGURES_DIR

import os, json, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import cv2 as cv
from PIL import Image
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import tensorflow as tf
import eval_edges_v2 as V

EREPO = EDGE_REPO
OUT = {}

# ---------- 1) quantization ablation ----------
imgs, gt_bins, gt_softs, files = V.load_pairs()

def ods_f(preds):
    best, bt = 0, None
    for t in V.THRESHOLDS:
        s = [0,0,0,0]
        for i in range(len(preds)):
            c = V.match_counts(preds[i] > t, gt_bins[i])
            s = [a+b for a,b in zip(s,c)]
        f = V.f_from_counts(*s)[0]
        if f > best: best, bt = f, t
    return round(best,3), round(float(bt),2)

def run_tflite(path, imgs):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]; out = interp.get_output_details()[0]
    preds, times = [], []
    for im in imgs:
        if inp['dtype'] == np.uint8:
            scale, zp = inp['quantization']
            x = ((im/scale + zp) if scale else im*255).astype(np.uint8) if scale else (im*255).astype(np.uint8)
            x = np.expand_dims(x, 0)
        else:
            x = np.expand_dims(im.astype(np.float32), 0)
        interp.set_tensor(inp['index'], x)
        t0 = time.perf_counter(); interp.invoke(); times.append(time.perf_counter()-t0)
        y = interp.get_tensor(out['index'])[0]
        if out['dtype'] == np.uint8:
            scale, zp = out['quantization']
            y = (y.astype(np.float32) - zp) * scale if scale else y.astype(np.float32)/255.0
        preds.append(np.squeeze(y).astype(np.float32))
    return preds, float(np.mean(times)*1000)

# float32 reference: which keras trial corresponds to the deployed tflite models?
km = tf.keras.models.load_model(f'{EREPO}/saved_models/cnn_trial_13.keras', compile=False)
fp = list(km.predict(imgs, batch_size=8, verbose=0)[...,0])
f32 = ods_f(fp)
OUT['edge_float32_trial13'] = {'ODS': f32[0], 'thresh': f32[1]}
print('float32 trial13 ODS', f32, flush=True)

for mn in [10, 11]:
    p = f'{EREPO}/lite/lite-models/model_{mn}.tflite'
    try:
        preds, ms = run_tflite(p, imgs)
        r = ods_f(preds)
        OUT[f'edge_tflite_model_{mn}'] = {'ODS': r[0], 'thresh': r[1],
                                          'x86_latency_ms': round(ms,1),
                                          'size_kb': round(os.path.getsize(p)/1024,1)}
        print(f'tflite model_{mn}:', OUT[f'edge_tflite_model_{mn}'], flush=True)
    except Exception as e:
        print(f'tflite model_{mn} ERROR: {str(e)[:200]}', flush=True)

# ---------- 2) dark channel prior baseline ----------
def dark_channel(im, sz=15):
    b,g,r = cv.split(im); dc = cv.min(cv.min(r,g),b)
    k = cv.getStructuringElement(cv.MORPH_RECT,(sz,sz))
    return cv.erode(dc,k)

def atm_light(im, dark):
    h,w = im.shape[:2]; n = h*w
    npx = max(n//1000, 1)
    idx = dark.reshape(n).argsort()[-npx:]
    return np.mean(im.reshape(n,3)[idx], axis=0)

def dcp_dehaze(im, omega=0.95, t0=0.1, sz=15):
    dark = dark_channel(im, sz); A = atm_light(im, dark)
    t = 1 - omega*dark_channel(im/A, sz)
    try:
        t = cv.ximgproc.guidedFilter((im*255).astype(np.uint8),
                                     (t*255).astype(np.uint8), 40, 1e-3).astype(np.float32)/255.0
    except Exception:
        t = cv.GaussianBlur(t.astype(np.float32), (41,41), 0)
    t = np.clip(t, t0, 1)
    return np.clip((im - A)/t[...,None] + A, 0, 1).astype(np.float32)

def load_dir(d):
    fs = sorted(os.listdir(d), key=lambda x: int(os.path.splitext(x)[0]))
    return np.stack([np.asarray(Image.open(os.path.join(d,f)).convert('RGB').resize((256,256), Image.BILINEAR),dtype=np.float32)/255.0 for f in fs])

hazy = load_dir(os.path.join(SMOKE_DATA, 'test', 'hazy'))
clean = load_dir(os.path.join(SMOKE_DATA, 'test', 'clean'))
t0 = time.perf_counter()
dcp = np.stack([dcp_dehaze(h) for h in hazy])
dcp_ms = (time.perf_counter()-t0)/len(hazy)*1000
ps = float(np.mean([psnr(clean[i], dcp[i], data_range=1.0) for i in range(len(clean))]))
ss = float(np.mean([ssim(clean[i], dcp[i], data_range=1.0, channel_axis=2) for i in range(len(clean))]))
OUT['dcp_baseline_smoke_test'] = {'PSNR': round(ps,2), 'SSIM': round(ss,3), 'x86_latency_ms': round(dcp_ms,1)}
print('DCP baseline:', OUT['dcp_baseline_smoke_test'], flush=True)

with open(os.path.join(RESULTS_DIR, 'results_extras.json'),'w') as f:
    json.dump(OUT, f, indent=2)
print(json.dumps(OUT, indent=2))
