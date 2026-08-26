"""DehazeNet on the full 165-pair corpus. Disk-backed data path.

Protocol identical to baselines_dehaze_fair.py: Adam 1e-3, batch 16, 50 epochs,
patience 5, restore_best_weights, trailing-20% validation split. The corpus is
staged to a uint8 memmap and converted per batch, because holding it in float32
OOM-killed the in-memory version. Images are 8-bit sources, so the round-trip
costs ~48 dB of quantization noise against a 13-18 dB signal.
"""
import os, sys, json, random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '/home/claude/code_release')
from config import SMOKE_DATA, RESULTS_DIR

import numpy as np, cv2 as cv
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

SIZE = (256, 256); SEED = 0; MULT = 7; BATCH = 16
CACHE = '/home/claude/experiments/_corpus165'
tf.random.set_seed(SEED); np.random.seed(SEED); random.seed(SEED)


def load(split, sub):
    d = f'{SMOKE_DATA}/{split}/{sub}'
    fs = sorted(os.listdir(d))
    out = np.empty((len(fs), *SIZE, 3), np.float32)
    for i, f in enumerate(fs):
        out[i] = np.asarray(Image.open(os.path.join(d, f)).convert('RGB')
                            .resize(SIZE, Image.BILINEAR), np.float32) / 255.0
    return out


Xr, Yr = load('train', 'hazy'), load('train', 'clean')
n = len(Xr); N = n * (MULT + 1)
print('pairs', n, '-> corpus', N, flush=True)

xp, yp = CACHE + '_x.npy', CACHE + '_y.npy'
if not os.path.exists(yp):
    Xm = np.lib.format.open_memmap(xp, 'w+', np.uint8, (N, *SIZE, 3))
    Ym = np.lib.format.open_memmap(yp, 'w+', np.uint8, (N, *SIZE, 3))
    q = lambda a: np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Xm[:n] = q(Xr); Ym[:n] = q(Yr)
    k = n
    for a, b in zip(Xr, Yr):
        for _ in range(MULT):
            u, v = a, b
            if random.random() > 0.5: u, v = np.fliplr(u), np.fliplr(v)
            if random.random() > 0.5: u, v = np.flipud(u), np.flipud(v)
            M = cv.getRotationMatrix2D((SIZE[1] // 2, SIZE[0] // 2), random.uniform(-180, 180), 1.0)
            Xm[k] = q(cv.warpAffine(u, M, SIZE, flags=cv.INTER_LINEAR))
            Ym[k] = q(cv.warpAffine(v, M, SIZE, flags=cv.INTER_LINEAR))
            k += 1
    Xm.flush(); Ym.flush(); del Xm, Ym
del Xr, Yr
Xall = np.load(xp, mmap_mode='r'); Yall = np.load(yp, mmap_mode='r')
print('corpus staged', Xall.shape, flush=True)

nv = int(N * 0.2); ntr = N - nv


class Seq(tf.keras.utils.Sequence):
    def __init__(self, lo, hi, shuffle):
        self.idx = np.arange(lo, hi); self.shuffle = shuffle
        if shuffle: np.random.shuffle(self.idx)
    def __len__(self): return int(np.ceil(len(self.idx) / BATCH))
    def __getitem__(self, i):
        j = np.sort(self.idx[i * BATCH:(i + 1) * BATCH])
        return (Xall[j].astype(np.float32) / 255.0, Yall[j].astype(np.float32) / 255.0)
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.idx)


def build_dehazenet():
    inp = layers.Input(shape=(*SIZE, 3))
    f = layers.Conv2D(16, 5, padding='same')(inp)
    def maxout(x, groups=4):
        sh = tf.shape(x)
        xr = tf.reshape(x, [sh[0], sh[1], sh[2], groups, x.shape[-1] // groups])
        return tf.reduce_max(xr, axis=3)
    f = layers.Lambda(lambda x: maxout(x, 4), output_shape=(*SIZE, 4))(f)
    m3 = layers.Conv2D(16, 3, padding='same', activation='relu')(f)
    m5 = layers.Conv2D(16, 5, padding='same', activation='relu')(f)
    m7 = layers.Conv2D(16, 7, padding='same', activation='relu')(f)
    m = layers.Concatenate()([m3, m5, m7])
    m = layers.MaxPooling2D(pool_size=7, strides=1, padding='same')(m)
    t = layers.Conv2D(1, 6, padding='same')(m)
    t = layers.Lambda(lambda x: tf.clip_by_value(x, 0.05, 1.0), output_shape=(*SIZE, 1))(t)
    def recon(z):
        t_, i_ = z
        A = tf.reduce_max(tf.nn.avg_pool2d(i_, 15, 1, 'SAME'), axis=[1, 2], keepdims=True)
        return tf.clip_by_value((i_ - A * (1.0 - t_)) / t_, 0.0, 1.0)
    return models.Model(inp, layers.Lambda(recon, output_shape=(*SIZE, 3))([t, inp]),
                        name='DehazeNet')


m = build_dehazenet()
m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
m.fit(Seq(0, ntr, True), epochs=50, validation_data=Seq(ntr, N, False), verbose=2,
      callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                                  restore_best_weights=True)])

Xte, Yte = load('test', 'hazy'), load('test', 'clean')
P = m.predict(Xte, batch_size=2, verbose=0)
ps = round(float(np.mean([psnr(Yte[i], P[i], data_range=1.0) for i in range(len(Yte))])), 2)
ss = round(float(np.mean([ssim(Yte[i], P[i], data_range=1.0, channel_axis=2) for i in range(len(Yte))])), 3)
print(f'DehazeNet: PSNR={ps} SSIM={ss} params={m.count_params()}', flush=True)

p = os.path.join(RESULTS_DIR, 'results_baselines_fair165.json')
prev = json.load(open(p)) if os.path.exists(p) else {}
prev['DehazeNet'] = {'PSNR': ps, 'SSIM': ss, 'params': int(m.count_params())}
prev['_corpus'] = {'train_pairs': n, 'augmented_images': N,
                   'note': 'SMOKE 110 + DENSE-HAZE 55; 7 augmented copies plus the original'}
json.dump(prev, open(p, 'w'), indent=2)
print(json.dumps(prev, indent=2))
