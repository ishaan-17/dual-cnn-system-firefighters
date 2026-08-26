"""Re-export both models with fully static shapes.

The Pi's tflite-runtime 2.11 rejects TILE v3, which appears because UpSampling2D
gets lowered to a dynamic-shape chain (SHAPE -> STRIDED_SLICE -> TILE) when the
batch dimension is unknown. Converting from a concrete function with a fixed
(1,256,256,3) signature should lower it to RESIZE_NEAREST_NEIGHBOR instead.

Also reports the op set of everything produced, so we can check it against what
2.11 supports before shipping it to the Pi again.
"""
import os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

EREPO = '/home/claude/dual-cnn-system-firefighters/edge'
SREPO = '/home/claude/dual-cnn-system-firefighters/smoke'
SM = '/home/claude/dual-cnn-system-firefighters/smoke/SMOKE'
OUT = '/home/claude/experiments/pi_kit'


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


def rep_from(dirpath, n=100):
    fs = sorted(os.listdir(dirpath))[:n]
    def gen():
        for f in fs:
            im = Image.open(os.path.join(dirpath, f)).convert('RGB').resize((256, 256), Image.BILINEAR)
            yield [np.expand_dims(np.asarray(im, dtype=np.float32) / 255.0, 0)]
    return gen


def convert_static(model, rep_gen, name):
    """Convert from a concrete function with a fixed batch dim."""
    @tf.function(input_signature=[tf.TensorSpec([1, 256, 256, 3], tf.float32)])
    def serve(x):
        return model(x, training=False)

    conc = serve.get_concrete_function()
    conv = tf.lite.TFLiteConverter.from_concrete_functions([conc], model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.uint8
    # keep the op set conservative for an old runtime
    conv._experimental_lower_tensor_list_ops = True
    buf = conv.convert()
    path = os.path.join(OUT, name)
    open(path, 'wb').write(buf)
    print(f'{name}: {len(buf)/1024:.1f} KB')
    print('   ops:', ops_of(buf), flush=True)
    return path, buf


results = {}

# ---------------- edge ----------------
edge = tf.keras.models.load_model(f'{EREPO}/saved_models/cnn_trial_13.keras', compile=False)
edge_rep = rep_from(f'{EREPO}/BIPEDv2/BIPED/edges/imgs/train/rgbr/real/')
p, buf = convert_static(edge, edge_rep, 'edge_int8_static.tflite')
results['edge'] = {'ops': [list(o) for o in ops_of(buf)], 'kb': round(len(buf)/1024, 1)}

# ---------------- dehazer ----------------
try:
    dehaze = tf.keras.models.load_model(f'{SREPO}/saved_models/dehazer_trial_8.keras', compile=False)
except Exception:
    dehaze = tf.keras.models.load_model(f'{SREPO}/saved_models/dehazer_trial_8.keras', compile=False, safe_mode=False)
dh_rep = rep_from(f'{SM}/train/hazy')
p2, buf2 = convert_static(dehaze, dh_rep, 'dehazer_int8_static.tflite')
results['dehazer'] = {'ops': [list(o) for o in ops_of(buf2)], 'kb': round(len(buf2)/1024, 1)}

json.dump(results, open('/home/claude/experiments/results_static_export.json', 'w'), indent=2)
print(json.dumps(results, indent=2))
