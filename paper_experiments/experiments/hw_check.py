"""What changes on a Raspberry Pi and what doesn't.

1) Accuracy invariance: does the quantized model produce identical output under
   different thread counts / execution paths? (proxy for platform independence of
   integer arithmetic)
2) Compute cost: MACs for each stage. Parameter count is a poor proxy for latency
   because the edge net runs at full 256x256 resolution while the dehazer downsamples.
   The paper's gating argument depends on which stage actually dominates compute.
3) x86 latency, single vs multi thread, per stage.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CNN_REPO, EDGE_REPO, SMOKE_REPO, SMOKE_DATA, RESULTS_DIR, FIGURES_DIR

import os, json, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

EREPO = EDGE_REPO
SREPO = SMOKE_REPO
OUT = {}

# ---------- 1) determinism of the quantized model across thread counts ----------
path = f'{EREPO}/lite/lite-models/model_10.tflite'
rng = np.random.default_rng(0)
x = rng.integers(0, 256, size=(1, 256, 256, 3), dtype=np.uint8)

outs = {}
for nthreads in [1, 2, 4]:
    interp = tf.lite.Interpreter(model_path=path, num_threads=nthreads)
    interp.allocate_tensors()
    i = interp.get_input_details()[0]; o = interp.get_output_details()[0]
    interp.set_tensor(i['index'], x)
    interp.invoke()
    outs[nthreads] = interp.get_tensor(o['index']).copy()

base = outs[1]
OUT['quantized_thread_invariance'] = {
    f'{n}_threads_max_abs_diff_vs_1thread': int(np.abs(outs[n].astype(int) - base.astype(int)).max())
    for n in outs
}
OUT['quantized_output_dtype'] = str(base.dtype)
print('thread invariance:', OUT['quantized_thread_invariance'], flush=True)

# ---------- 2) MAC counts ----------
def macs_of(model):
    """Analytic MACs for conv / separable conv / transposed conv / dense layers."""
    total = 0
    per_layer = []
    for L in model.layers:
        cls = L.__class__.__name__
        try:
            out_shape = L.output.shape
        except Exception:
            continue
        if len(out_shape) != 4:
            continue
        _, oh, ow, oc = out_shape
        if oh is None or ow is None:
            continue
        m = 0
        if cls == 'Conv2D':
            kh, kw = L.kernel_size
            ic = L.input.shape[-1]
            m = kh * kw * ic * oc * oh * ow
        elif cls == 'SeparableConv2D':
            kh, kw = L.kernel_size
            ic = L.input.shape[-1]
            # depthwise over the INPUT spatial size, pointwise over the output
            ih, iw = L.input.shape[1], L.input.shape[2]
            m = kh * kw * ic * ih * iw + ic * oc * oh * ow
        elif cls == 'DepthwiseConv2D':
            kh, kw = L.kernel_size
            ic = L.input.shape[-1]
            m = kh * kw * ic * oh * ow
        elif cls == 'Conv2DTranspose':
            kh, kw = L.kernel_size
            ic = L.input.shape[-1]
            ih, iw = L.input.shape[1], L.input.shape[2]
            m = kh * kw * ic * oc * ih * iw
        if m:
            total += m
            per_layer.append((L.name, cls, int(m)))
    return total, per_layer

edge = tf.keras.models.load_model(f'{EREPO}/saved_models/cnn_trial_13.keras', compile=False)
try:
    dehaze = tf.keras.models.load_model(f'{SREPO}/saved_models/dehazer_trial_8.keras', compile=False)
except Exception:
    dehaze = tf.keras.models.load_model(f'{SREPO}/saved_models/dehazer_trial_8.keras', compile=False, safe_mode=False)

e_macs, e_layers = macs_of(edge)
d_macs, d_layers = macs_of(dehaze)
OUT['macs'] = {
    'edge_trial13': int(e_macs),
    'dehazer_trial8': int(d_macs),
    'dehazer_over_edge': round(d_macs / e_macs, 2),
    'edge_params': int(edge.count_params()),
    'dehazer_params': int(dehaze.count_params()),
    'dehazer_params_over_edge': round(dehaze.count_params() / edge.count_params(), 2),
}
print('MACs:', OUT['macs'], flush=True)

# ---------- 3) x86 latency per stage, single vs multi thread ----------
def bench_keras(model, n=12, warmup=3):
    xb = np.zeros((1, 256, 256, 3), dtype=np.float32)
    f = tf.function(lambda t: model(t, training=False))
    for _ in range(warmup): f(xb)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(xb); ts.append(time.perf_counter() - t0)
    return round(float(np.median(ts)) * 1000, 1)

def bench_tflite(path, nthreads, n=20, warmup=5):
    interp = tf.lite.Interpreter(model_path=path, num_threads=nthreads)
    interp.allocate_tensors()
    i = interp.get_input_details()[0]
    xb = np.zeros(i['shape'], dtype=i['dtype'])
    for _ in range(warmup):
        interp.set_tensor(i['index'], xb); interp.invoke()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        interp.set_tensor(i['index'], xb); interp.invoke()
        ts.append(time.perf_counter() - t0)
    return round(float(np.median(ts)) * 1000, 1)

OUT['x86_latency_ms'] = {
    'edge_tflite_uint8_1thread': bench_tflite(path, 1),
    'edge_tflite_uint8_4thread': bench_tflite(path, 4),
    'edge_keras_float32': bench_keras(edge),
    'dehazer_keras_float32': bench_keras(dehaze),
}
print('x86 latency:', OUT['x86_latency_ms'], flush=True)

json.dump(OUT, open(os.path.join(RESULTS_DIR, 'results_hw.json'), 'w'), indent=2)
print(json.dumps(OUT, indent=2))
