#!/usr/bin/env python3
"""Raspberry Pi 4 benchmark for the dual-CNN firefighter pipeline.

Measures what the paper needs and nothing else:
  (a) edge-only            latency / FPS
  (b) dehaze + edge        latency / FPS
  (c) gated pipeline       latency / FPS at several haze levels
  plus CPU load, temperature, and thermal drift over a sustained run.

USAGE ON THE PI
---------------
  sudo apt install -y python3-opencv python3-numpy
  pip3 install tflite-runtime            # or use full tensorflow if already installed

  python3 pi_benchmark.py \
      --edge   model_10.tflite \
      --dehaze dehazer_int8.tflite \
      --iters 100

  # edge stage only (if you have not exported the dehazer yet):
  python3 pi_benchmark.py --edge model_10.tflite --iters 100

Paste the printed JSON block back and it goes straight into the paper.

NOTES
-----
* Use model_10.tflite specifically. edge/lite/run-quant-demo.py defaults to
  model_8.tflite, which has the same weights but is not the file scored in the
  paper's table.
* Run on mains power, not a battery, and leave the Pi idle for ~2 min first so
  you start from a cool baseline.
* --threads defaults to 4 (all Cortex-A72 cores). Also try --threads 1 to report
  a single-core number.
"""
import argparse, json, os, statistics, subprocess, time

import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter

SIZE = 256
A_LIGHT = 0.8


# ---------------------------------------------------------------- utilities
def cpu_temp_c():
    """Pi SoC temperature in Celsius, or None if unavailable."""
    for path in ('/sys/class/thermal/thermal_zone0/temp',):
        try:
            with open(path) as fh:
                return round(int(fh.read().strip()) / 1000.0, 1)
        except Exception:
            pass
    try:
        out = subprocess.check_output(['vcgencmd', 'measure_temp'], text=True)
        return float(out.strip().split('=')[1].rstrip("'C"))
    except Exception:
        return None


def throttled_flags():
    """Pi throttling bitmask; nonzero means the clock was capped at some point."""
    try:
        out = subprocess.check_output(['vcgencmd', 'get_throttled'], text=True)
        return out.strip().split('=')[1]
    except Exception:
        return None


def cpu_percent(interval=0.5):
    """Whole-machine CPU utilisation from /proc/stat, no psutil dependency."""
    def snap():
        with open('/proc/stat') as fh:
            parts = [float(x) for x in fh.readline().split()[1:]]
        idle = parts[3] + parts[4]
        return sum(parts), idle
    t0, i0 = snap()
    time.sleep(interval)
    t1, i1 = snap()
    dt, di = t1 - t0, i1 - i0
    return round(100.0 * (1.0 - di / dt), 1) if dt > 0 else None


def load_interp(path, threads):
    it = Interpreter(model_path=path, num_threads=threads)
    it.allocate_tensors()
    return it, it.get_input_details()[0], it.get_output_details()[0]


def run_once(interp, inp, out, x):
    interp.set_tensor(inp['index'], x)
    interp.invoke()
    return interp.get_tensor(out['index'])


def as_input(x_float, detail):
    """Cast a float [0,1] HWC frame to whatever the interpreter expects."""
    if detail['dtype'] == np.uint8:
        return np.expand_dims((x_float * 255).astype(np.uint8), 0)
    return np.expand_dims(x_float.astype(np.float32), 0)


def timeit(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return {
        'median_ms': round(statistics.median(ts), 2),
        'mean_ms': round(statistics.mean(ts), 2),
        'p95_ms': round(ts[int(0.95 * (len(ts) - 1))], 2),
        'min_ms': round(ts[0], 2),
        'max_ms': round(ts[-1], 2),
        'fps_from_median': round(1000.0 / statistics.median(ts), 2),
    }


def dark_channel_mean(img, patch=15):
    """The paper's gate statistic: mean of the dark channel."""
    import cv2 as cv
    dc = img.min(axis=2)
    k = cv.getStructuringElement(cv.MORPH_RECT, (patch, patch))
    return float(cv.erode(dc, k).mean())


# ---------------------------------------------------------------- benchmark
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edge', required=True, help='path to the uint8 edge .tflite (use model_10.tflite)')
    ap.add_argument('--dehaze', default=None, help='path to the dehazer .tflite, if exported')
    ap.add_argument('--iters', type=int, default=100)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--tau', type=float, default=None,
                    help='gate threshold; defaults to the calibrated value printed by the paper')
    ap.add_argument('--soak-seconds', type=int, default=120,
                    help='sustained run for thermal drift; 0 to skip')
    args = ap.parse_args()

    # Use a real photograph, not noise: the gate's statistic is a dark-channel
    # mean, and uniform random noise has a dark channel near zero, so the gate
    # would never fire and the gated timings would only ever measure the skip
    # branch. Falls back to noise (with a warning) if the sample is missing.
    if os.path.exists('sample_frame.png'):
        import cv2 as _cv
        _img = _cv.imread('sample_frame.png')
        _img = _cv.cvtColor(_img, _cv.COLOR_BGR2RGB)
        frame = _cv.resize(_img, (SIZE, SIZE)).astype(np.float32) / 255.0
    else:
        print('WARNING: sample_frame.png missing; gated timings will be invalid', flush=True)
        rng = np.random.default_rng(0)
        frame = rng.random((SIZE, SIZE, 3), dtype=np.float32)

    res = {
        'device': os.uname().machine,
        'threads': args.threads,
        'iters': args.iters,
        'temp_start_c': cpu_temp_c(),
        'throttled_start': throttled_flags(),
    }
    try:
        with open('/proc/device-tree/model') as fh:
            res['model'] = fh.read().strip('\x00')
    except Exception:
        pass

    # ---- (a) edge only
    e_it, e_in, e_out = load_interp(args.edge, args.threads)
    xe = as_input(frame, e_in)
    res['edge_only'] = timeit(lambda: run_once(e_it, e_in, e_out, xe), args.iters, args.warmup)
    res['edge_only']['cpu_percent'] = cpu_percent()
    res['edge_only']['temp_c'] = cpu_temp_c()
    print('edge-only done', flush=True)

    # ---- (b) dehaze + edge
    if args.dehaze:
        d_it, d_in, d_out = load_interp(args.dehaze, args.threads)
        xd = as_input(frame, d_in)

        res['dehaze_only'] = timeit(lambda: run_once(d_it, d_in, d_out, xd), args.iters, args.warmup)
        res['dehaze_only']['temp_c'] = cpu_temp_c()

        def full():
            y = run_once(d_it, d_in, d_out, xd)
            y = np.squeeze(y)
            if y.dtype == np.uint8:
                yf = y.astype(np.float32) / 255.0
            else:
                yf = y
            run_once(e_it, e_in, e_out, as_input(yf, e_in))

        res['dehaze_plus_edge'] = timeit(full, args.iters, args.warmup)
        res['dehaze_plus_edge']['cpu_percent'] = cpu_percent()
        res['dehaze_plus_edge']['temp_c'] = cpu_temp_c()
        print('dehaze+edge done', flush=True)

        # ---- (c) gated, at three haze levels
        tau = args.tau
        if tau is None:
            tau = 0.35   # replace with the calibrated tau reported in the paper
            res['tau_note'] = 'default placeholder; pass --tau with the calibrated value'
        res['tau'] = tau

        gated = {}
        for t in (0.7, 0.5, 0.35):
            hazy = np.clip(frame * t + A_LIGHT * (1 - t), 0, 1).astype(np.float32)
            xh_e = as_input(hazy, e_in)
            xh_d = as_input(hazy, d_in)

            def gated_run():
                score = dark_channel_mean(hazy)
                if score >= tau:
                    y = np.squeeze(run_once(d_it, d_in, d_out, xh_d))
                    yf = y.astype(np.float32) / 255.0 if y.dtype == np.uint8 else y
                    run_once(e_it, e_in, e_out, as_input(yf, e_in))
                else:
                    run_once(e_it, e_in, e_out, xh_e)

            g = timeit(gated_run, args.iters, args.warmup)
            g['haze_score'] = round(dark_channel_mean(hazy), 4)
            g['dehazed'] = bool(g['haze_score'] >= tau)
            gated[f't={t}'] = g
            print(f'gated t={t} done', flush=True)
        res['gated'] = gated

        # gate statistic cost on its own
        res['gate_statistic'] = timeit(lambda: dark_channel_mean(frame), 200, 20)
    else:
        res['note'] = 'dehazer .tflite not supplied; only edge-stage numbers measured'

    # ---- sustained run for thermal behaviour
    if args.soak_seconds > 0:
        t_end = time.time() + args.soak_seconds
        n, temps = 0, []
        while time.time() < t_end:
            run_once(e_it, e_in, e_out, xe)
            n += 1
            if n % 50 == 0:
                temps.append(cpu_temp_c())
        res['soak'] = {
            'seconds': args.soak_seconds,
            'frames': n,
            'sustained_fps': round(n / args.soak_seconds, 2),
            'temp_series_c': temps,
            'temp_end_c': cpu_temp_c(),
            'throttled_end': throttled_flags(),
        }
        print('soak done', flush=True)

    print('\n================ PASTE EVERYTHING BELOW ================')
    print(json.dumps(res, indent=2))
    print('================ END ================')


if __name__ == '__main__':
    main()
