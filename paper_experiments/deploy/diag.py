#!/usr/bin/env python3
"""Isolate a segfault. Run:  python3 diag.py

Tests one thing at a time and flushes after each line, so whatever prints LAST
is the step that crashed.
"""
import sys, platform

def say(s):
    print(s, flush=True)

say("=== environment ===")
say(f"python   : {sys.version.split()[0]}")
say(f"machine  : {platform.machine()}")

try:
    import numpy as np
    say(f"numpy    : {np.__version__}")
    if int(np.__version__.split('.')[0]) >= 2:
        say("  !! numpy 2.x with tflite-runtime 2.11 is a known segfault cause.")
        say("     fix: pip3 install 'numpy<2'")
except Exception as e:
    say(f"numpy    : FAILED {e}")

try:
    import tflite_runtime
    say(f"tflite   : {tflite_runtime.__version__}")
    from tflite_runtime.interpreter import Interpreter
except Exception:
    say("tflite   : not found, trying tensorflow")
    from tensorflow.lite import Interpreter

import numpy as np

def test(path, label, threads):
    say(f"\n=== {label}  (threads={threads}) ===")
    say("  allocating...")
    it = Interpreter(model_path=path, num_threads=threads)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    say(f"  input  {i['shape']} {i['dtype'].__name__}")
    say(f"  output {o['shape']} {o['dtype'].__name__}")
    x = np.zeros(i['shape'], dtype=i['dtype'])
    say("  invoking...")
    it.set_tensor(i['index'], x)
    it.invoke()
    y = it.get_tensor(o['index'])
    say(f"  OK -> output range {float(y.min()):.3f}..{float(y.max()):.3f}")

# single-threaded first: threading is itself a common segfault source
for threads in (1, 4):
    try:
        test("edge_int8.tflite", "EDGE", threads)
    except Exception as e:
        say(f"  EDGE FAILED: {type(e).__name__}: {e}")
    try:
        test("dehazer_int8.tflite", "DEHAZER", threads)
    except Exception as e:
        say(f"  DEHAZER FAILED: {type(e).__name__}: {e}")

say("\n=== all tests completed without segfault ===")
