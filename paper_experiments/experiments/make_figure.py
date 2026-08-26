"""Gate figure: ODS vs haze density for the three execution policies.
All values strict one-to-one protocol, deployed 23K edge model.
The gated policy tracks the upper envelope of the two fixed policies.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CNN_REPO, EDGE_REPO, SMOKE_REPO, SMOKE_DATA, RESULTS_DIR, FIGURES_DIR

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import json

BLUE, ORANGE, GREEN = '#1f5fa9', '#d1662b', '#2f7d4f'

d = json.load(open(os.path.join(RESULTS_DIR, 'results_gate.json')))['per_level']
order = ['clean', 't=0.7', 't=0.5', 't=0.35']
x      = [0.0, 0.30, 0.50, 0.65]           # haze density 1-t
never  = [d[k]['never_dehaze']  for k in order]
always = [d[k]['always_dehaze'] for k in order]
gated  = [d[k]['gated']         for k in order]

with plt.style.context(['science', 'no-latex']):
    fig, ax = plt.subplots(figsize=(2.7, 2.15))
    ax.plot(x, never,  color=ORANGE, marker='o', ms=3.6, lw=1.2, label='never dehaze', zorder=3)
    ax.plot(x, always, color=BLUE,   marker='s', ms=3.6, lw=1.2, label='always dehaze', zorder=3)
    ax.plot(x, gated,  color=GREEN,  marker='^', ms=4.2, lw=1.6, ls=(0, (5, 1.6)),
            label='gated (ours)', zorder=4)
    ax.set_xlabel('haze density  $1-t$')
    ax.set_ylabel('edge F-measure (ODS)')
    ax.set_xlim(-0.04, 0.69)
    ax.set_ylim(0.41, 0.775)
    ax.set_xticks(x)
    ax.set_xticklabels(['0', '0.30', '0.50', '0.65'])
    ax.set_yticks([0.45, 0.55, 0.65, 0.75])
    ax.tick_params(labelsize=7.5)
    ax.xaxis.label.set_size(8.5); ax.yaxis.label.set_size(8.5)
    ax.legend(loc='lower left', fontsize=6.8, frameon=False,
              handlelength=1.7, borderpad=0.2, labelspacing=0.3)
    fig.savefig(os.path.join(FIGURES_DIR, 'gate.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'gate.png'), dpi=300, bbox_inches='tight')

print('saved gate figure')
print('never ', never)
print('always', always)
print('gated ', gated)
