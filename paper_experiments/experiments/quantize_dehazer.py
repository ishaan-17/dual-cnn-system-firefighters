"""Export the dehazer to full-integer TFLite and verify what quantization costs.

Answers the reviewer's 'the dehazer is not quantized' directly, and produces the
dehazer_int8.tflite that pi_benchmark.py needs for end-to-end timing.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CNN_REPO, EDGE_REPO, SMOKE_REPO, SMOKE_DATA, RESULTS_DIR, FIGURES_DIR

import os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from PIL import Image
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

SREPO = SMOKE_REPO
SM = SMOKE_DATA
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'deploy', 'dehazer_int8.tflite')
SIZE = (256, 256)


def load(split, sub):
    d = f'{SM}/{split}/{sub}'
    fs = sorted(os.listdir(d), key=lambda x: int(os.path.splitext(x)[0]))
    return np.stack([np.asarray(Image.open(os.path.join(d, f)).convert('RGB').resize(SIZE, Image.BILINEAR),
                                dtype=np.float32) / 255.0 for f in fs])


Xtr = load('train', 'hazy')
Xte = load('test', 'hazy')
Yte = load('test', 'clean')

try:
    model = tf.keras.models.load_model(f'{SREPO}/saved_models/dehazer_trial_8.keras', compile=False)
except Exception:
    model = tf.keras.models.load_model(f'{SREPO}/saved_models/dehazer_trial_8.keras', compile=False, safe_mode=False)


def rep_data():
    # real training frames, not noise: calibration quality matters for activations
    for i in range(min(100, len(Xtr))):
        yield [np.expand_dims(Xtr[i], 0).astype(np.float32)]


conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_data
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.uint8
conv.inference_output_type = tf.uint8
tfl = conv.convert()
open(OUT_PATH, 'wb').write(tfl)
size_kb = round(os.path.getsize(OUT_PATH) / 1024, 1)
print(f'wrote {OUT_PATH} ({size_kb} KB)', flush=True)

# ---- accuracy of the quantized dehazer vs float32
it = tf.lite.Interpreter(model_path=OUT_PATH)
it.allocate_tensors()
inp, out = it.get_input_details()[0], it.get_output_details()[0]

qp = []
for i in range(len(Xte)):
    x = np.expand_dims((Xte[i] * 255).astype(np.uint8), 0)
    it.set_tensor(inp['index'], x); it.invoke()
    y = it.get_tensor(out['index'])[0]
    sc, zp = out['quantization']
    y = (y.astype(np.float32) - zp) * sc if sc else y.astype(np.float32) / 255.0
    qp.append(np.clip(y, 0, 1))
qp = np.stack(qp)

fp = model.predict(Xte, batch_size=4, verbose=0)

def score(P):
    return (round(float(np.mean([psnr(Yte[i], P[i], data_range=1.0) for i in range(len(Yte))])), 2),
            round(float(np.mean([ssim(Yte[i], P[i], data_range=1.0, channel_axis=2) for i in range(len(Yte))])), 3))

f_ps, f_ss = score(fp)
q_ps, q_ss = score(qp)
res = {
    'size_kb': size_kb,
    'float32': {'PSNR': f_ps, 'SSIM': f_ss},
    'uint8': {'PSNR': q_ps, 'SSIM': q_ss},
    'delta_PSNR': round(q_ps - f_ps, 2),
    'delta_SSIM': round(q_ss - f_ss, 3),
}
print(json.dumps(res, indent=2))
json.dump(res, open(os.path.join(RESULTS_DIR, 'results_quant_dehazer.json'), 'w'), indent=2)
