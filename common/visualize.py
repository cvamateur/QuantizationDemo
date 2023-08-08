import matplotlib.pyplot as plt
import numpy as np


import quantization as q

def plot_weight_distribution(weights, name, bitwidth=32):

    if bitwidth <= 8:
        qmin, qmax = q.Q_MIN(bitwidth), q.Q_MAX(bitwidth)
        bins = np.arange(qmin, qmax + 2)
        align = "left"
    else:
        bins = min(128, int(weights.numel() * 0.8))
        align = "mid"

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(weights.detach().cpu().view(-1), bins=bins, density=True,
            align=align, color='b', alpha=0.5, edgecolor='b' if bitwidth <= 4 else None)
    ax.set_xlabel(name)
    ax.set_ylabel("density")
    fig.suptitle(f'Histogram of Weights (bitwidth={bitwidth} bits)')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()
