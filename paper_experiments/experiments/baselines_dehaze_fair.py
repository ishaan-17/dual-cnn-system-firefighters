"""Small-model dehazing baselines trained on the same SMOKE split as ours.

AOD-Net (Li et al., ICCV 2017): estimates a single K map, reconstructs via the
  reformulated atmospheric model J = K*I - K + b.
DehazeNet (Cai et al., TIP 2016): feature extraction with maxout, multi-scale
  conv, max-pooling, BReLU; predicts transmission t, then J = (I - A(1-t))/t.

Both trained with the same optimizer, schedule, and data as our model so the
comparison isolates architecture rather than training budget.
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
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

SM = SMOKE_DATA
SIZE = (256, 256)
SEED = 0
tf.random.set_seed(SEED); np.random.seed(SEED)


def load(split):
    def rd(sub):
        d = f'{SM}/{split}/{sub}'
        fs = sorted(os.listdir(d))
        return np.stack([np.asarray(Image.open(os.path.join(d, f)).convert('RGB').resize(SIZE, Image.BILINEAR),
                                    dtype=np.float32) / 255.0 for f in fs])
    return rd('hazy'), rd('clean')


import random, cv2 as cv
random.seed(SEED)

def augment_pair(a, b):
    if random.random() > 0.5:
        a, b = np.fliplr(a), np.fliplr(b)
    if random.random() > 0.5:
        a, b = np.flipud(a), np.flipud(b)
    h, w = a.shape[:2]
    M = cv.getRotationMatrix2D((w // 2, h // 2), random.uniform(-180, 180), 1.0)
    return (cv.warpAffine(a, M, (w, h), flags=cv.INTER_LINEAR),
            cv.warpAffine(b, M, (w, h), flags=cv.INTER_LINEAR))

def augment(X, Y, multiple=7):
    ax, ay = [], []
    for a, b in zip(X, Y):
        for _ in range(multiple):
            u, v = augment_pair(a, b)
            ax.append(u); ay.append(v)
    return np.concatenate([X, np.array(ax)]), np.concatenate([Y, np.array(ay)])

Xtr, Ytr = load('train')
Xte, Yte = load('test')
Xtr, Ytr = augment(Xtr, Ytr, multiple=7)
print('train (augmented)', Xtr.shape, 'test', Xte.shape, flush=True)


# ---------------- AOD-Net ----------------
def build_aodnet():
    inp = layers.Input(shape=(*SIZE, 3))
    c1 = layers.Conv2D(3, 1, padding='same', activation='relu')(inp)
    c2 = layers.Conv2D(3, 3, padding='same', activation='relu')(c1)
    cc1 = layers.Concatenate()([c1, c2])
    c3 = layers.Conv2D(3, 5, padding='same', activation='relu')(cc1)
    cc2 = layers.Concatenate()([c2, c3])
    c4 = layers.Conv2D(3, 7, padding='same', activation='relu')(cc2)
    cc3 = layers.Concatenate()([c1, c2, c3, c4])
    K = layers.Conv2D(3, 3, padding='same', activation='relu')(cc3)
    # J = K*I - K + b, b = 1
    out = layers.Lambda(lambda z: tf.clip_by_value(z[0] * z[1] - z[0] + 1.0, 0.0, 1.0),
                        output_shape=(*SIZE, 3))([K, inp])
    return models.Model(inp, out, name='AOD-Net')


# ---------------- DehazeNet ----------------
def build_dehazenet():
    inp = layers.Input(shape=(*SIZE, 3))
    # feature extraction: 16 filters 5x5 -> maxout over 4 groups of 4
    f = layers.Conv2D(16, 5, padding='same')(inp)
    def maxout(x, groups=4):
        sh = tf.shape(x)
        xr = tf.reshape(x, [sh[0], sh[1], sh[2], groups, x.shape[-1] // groups])
        return tf.reduce_max(xr, axis=3)
    f = layers.Lambda(lambda x: maxout(x, 4), output_shape=(*SIZE, 4))(f)
    # multi-scale mapping
    m3 = layers.Conv2D(16, 3, padding='same', activation='relu')(f)
    m5 = layers.Conv2D(16, 5, padding='same', activation='relu')(f)
    m7 = layers.Conv2D(16, 7, padding='same', activation='relu')(f)
    m = layers.Concatenate()([m3, m5, m7])
    # local extremum (max pool, stride 1) then BReLU -> transmission map
    m = layers.MaxPooling2D(pool_size=7, strides=1, padding='same')(m)
    t = layers.Conv2D(1, 6, padding='same')(m)
    t = layers.Lambda(lambda x: tf.clip_by_value(x, 0.05, 1.0), output_shape=(*SIZE, 1))(t)
    # J = (I - A(1-t))/t with A estimated as the brightest 0.1% of the input
    def recon(z):
        t_, i_ = z
        A = tf.reduce_max(tf.nn.avg_pool2d(i_, 15, 1, 'SAME'), axis=[1, 2], keepdims=True)
        return tf.clip_by_value((i_ - A * (1.0 - t_)) / t_, 0.0, 1.0)
    out = layers.Lambda(recon, output_shape=(*SIZE, 3))([t, inp])
    return models.Model(inp, out, name='DehazeNet')


def evaluate(model):
    P = model.predict(Xte, batch_size=4, verbose=0)
    ps = float(np.mean([psnr(Yte[i], P[i], data_range=1.0) for i in range(len(Yte))]))
    ss = float(np.mean([ssim(Yte[i], P[i], data_range=1.0, channel_axis=2) for i in range(len(Yte))]))
    return round(ps, 2), round(ss, 3)


OUT = {}
for name, builder in [('AOD-Net', build_aodnet), ('DehazeNet', build_dehazenet)]:
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED); np.random.seed(SEED)
    m = builder()
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    cb = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    m.fit(Xtr, Ytr, epochs=50, batch_size=16, validation_split=0.2, callbacks=cb, verbose=0)
    ps, ss = evaluate(m)
    OUT[name] = {'PSNR': ps, 'SSIM': ss, 'params': int(m.count_params())}
    print(f'{name}: PSNR={ps} SSIM={ss} params={m.count_params()}', flush=True)

json.dump(OUT, open(os.path.join(RESULTS_DIR, 'results_baselines_fair.json'), 'w'), indent=2)
print(json.dumps(OUT, indent=2))
