"""Single place to point the experiments at your data.

Override with environment variables, or edit the defaults below.

  export CNN_REPO=/path/to/dual-cnn-system-firefighters
  export SMOKE_DATA=/path/to/SMOKE          # optional; defaults to the copy in CNN_REPO
  export RESULTS_DIR=/path/to/results

Expected layout under CNN_REPO (this is the published repo, unchanged):
  edge/saved_models/cnn_trial_{7,13}.keras
  edge/lite/lite-models/model_10.tflite
  edge/BIPEDv2/BIPED/edges/{imgs,edge_maps}/{train,test}/...
  smoke/saved_models/dehazer_trial_8.keras
  smoke/SMOKE/{train,test}/{hazy,clean}/
"""
import os

CNN_REPO    = os.environ.get('CNN_REPO', '/home/claude/dual-cnn-system-firefighters')
EDGE_REPO   = os.path.join(CNN_REPO, 'edge')
SMOKE_REPO  = os.path.join(CNN_REPO, 'smoke')
SMOKE_DATA  = os.environ.get('SMOKE_DATA', os.path.join(SMOKE_REPO, 'SMOKE'))
RESULTS_DIR = os.environ.get('RESULTS_DIR',
                             os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'))
FIGURES_DIR = os.environ.get('FIGURES_DIR',
                             os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures'))
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
