"""Rebuild the edge model with a static nearest-neighbour resize instead of
UpSampling2D, transfer the trained weights, and re-export to int8 TFLite.

Why: UpSampling2D lowers to a dynamic-shape chain (SHAPE -> STRIDED_SLICE ->
TILE v3) that tflite-runtime 2.11 on the Pi cannot register. tf.image.resize with
literal output sizes lowers to RESIZE_NEAREST_NEIGHBOR, which 2.11 supports.

Neither UpSampling2D nor the replacement has trainable weights, so the ordered
weight list is unchanged and set_weights() transfers exactly.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CNN_REPO, EDGE_REPO, SMOKE_REPO, SMOKE_DATA, RESULTS_DIR, FIGURES_DIR

import os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.lite.python import schema_py_generated as schema

EREPO = EDGE_REPO
SREPO = SMOKE_REPO
SM = SMOKE_DATA
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'deploy')


def ops_of(buf):
    m = schema.Model.GetRootAsModel(buf, 0)
    res = []
    for i in range(m.OperatorCodesLength()):
        oc = m.OperatorCodes(i)
        bc = oc.BuiltinCode() or oc.DeprecatedBuiltinCode()
        name = next((k for k, v in vars(schema.BuiltinOperator).items()
                     if not k.startswith('_') and v == bc), f'code{bc}')
        res.append((name, oc.Version()))
    return sorted(set(res))


class StaticResize(layers.Layer):
    """Nearest-neighbour upsample to a literal size -> RESIZE_NEAREST_NEIGHBOR."""
    def __init__(self, size, **kw):
        super().__init__(**kw)
        self.size = tuple(int(s) for s in size)

    def call(self, x):
        return tf.image.resize(x, self.size, method='nearest')

    def compute_output_shape(self, s):
        return (s[0], self.size[0], self.size[1], s[-1])

    def get_config(self):
        c = super().get_config(); c['size'] = self.size; return c


def build_edge(input_shape=(256, 256, 3)):
    """Same topology as edge/main.py cnn_model(), static resize in the decoder."""
    inputs = layers.Input(shape=input_shape, batch_size=1)

    def enc(x, filters, strides=(2, 2), k=(3, 3)):
        skip = layers.SeparableConv2D(filters, k, padding='same')(x)
        skip = layers.BatchNormalization()(skip)
        skip = layers.Activation('relu')(skip)
        y = layers.SeparableConv2D(filters, k, strides=strides, padding='same')(skip)
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        return y, skip

    def dec(x, skip, filters, size, k=(3, 3)):
        x = StaticResize(size)(x)
        x = layers.concatenate([x, skip])
        x = layers.SeparableConv2D(filters, k, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    x = inputs
    x, s1 = enc(x, 32)          # 128
    x, s2 = enc(x, 64)          # 64
    d2 = dec(x, s2, 64, (128, 128))
    d3 = dec(d2, s1, 32, (256, 256))
    outputs = layers.SeparableConv2D(1, (1, 1), activation='sigmoid', padding='same')(d3)
    return models.Model(inputs, outputs, name='EdgeStatic')


def rep_from(dirpath, n=100):
    fs = sorted(os.listdir(dirpath))[:n]
    def gen():
        for f in fs:
            im = Image.open(os.path.join(dirpath, f)).convert('RGB').resize((256, 256), Image.BILINEAR)
            yield [np.expand_dims(np.asarray(im, dtype=np.float32) / 255.0, 0)]
    return gen


def to_int8(model, rep, name):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.uint8
    buf = conv.convert()
    open(os.path.join(OUT, name), 'wb').write(buf)
    o = ops_of(buf)
    print(f'{name}: {len(buf)/1024:.1f} KB')
    print('   ops:', o, flush=True)
    return buf, o


# ---- rebuild edge + transfer weights ----
src = tf.keras.models.load_model(f'{EREPO}/saved_models/cnn_trial_13.keras', compile=False)
dst = build_edge()
sw, dw = src.get_weights(), dst.get_weights()
assert len(sw) == len(dw), f'weight count differs: {len(sw)} vs {len(dw)}'
for a, b in zip(sw, dw):
    assert a.shape == b.shape, f'shape mismatch {a.shape} vs {b.shape}'
dst.set_weights(sw)
print('weights transferred:', len(sw), 'tensors', flush=True)

# sanity: outputs must match the original
probe = np.random.default_rng(0).random((1, 256, 256, 3)).astype('float32')
d = float(np.abs(src.predict(probe, verbose=0) - dst.predict(probe, verbose=0)).max())
print(f'max |orig - rebuilt| = {d:.2e}', flush=True)

res = {'weight_match_maxdiff': d}
_, res['edge_ops'] = to_int8(dst, rep_from(f'{EREPO}/BIPEDv2/BIPED/edges/imgs/train/rgbr/real/'),
                             'edge_int8.tflite')

# ---- dehazer: already TILE-free, but re-check its ops ----
try:
    dh = tf.keras.models.load_model(f'{SREPO}/saved_models/dehazer_trial_8.keras', compile=False)
except Exception:
    dh = tf.keras.models.load_model(f'{SREPO}/saved_models/dehazer_trial_8.keras', compile=False, safe_mode=False)
_, res['dehazer_ops'] = to_int8(dh, rep_from(f'{SM}/train/hazy'), 'dehazer_int8.tflite')

json.dump({k: ([list(x) for x in v] if isinstance(v, list) else v) for k, v in res.items()},
          open(os.path.join(RESULTS_DIR, 'results_notile.json'), 'w'), indent=2)
print('done')
