import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ..functional import Q_MIN, Q_MAX
from ..q_types import t_Tensor, t_ndarray

def print_styles():
    from pprint import pprint
    pprint(matplotlib.style.available)


def plot_tensor_histogram(weights, name, bitwidth=32, sign: bool = True):
    if bitwidth <= 8:
        qmin, qmax = Q_MIN(bitwidth, sign), Q_MAX(bitwidth, sign)
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


def plot_tensor_statistics(tensor,
                           name: str,
                           dim: int = None,
                           percentile: float = 100.,
                           show_fliers: bool = False,
                           style: str = "seaborn-v0_8"):

    if isinstance(tensor, t_Tensor):
        tensor = tensor.detach().cpu().numpy()
    assert isinstance(tensor, t_ndarray), "error: input must be numpy.array"

    matplotlib.style.use(style)
    kwargs = {
        "showfliers": show_fliers,
        "flierprops": {"color": "C0", "linewidth": 0.5},
        "whis": (
            min(percentile, 100. - percentile),
            max(percentile, 100. - percentile)
        ),
    }

    if dim is None:
        num_channels = 1
        tensor = tensor.reshape(-1)
        labels = None
    else:
        num_channels = tensor.shape[dim]
        tensor = np.swapaxes(tensor, dim, -1)
        tensor = tensor.reshape(-1, num_channels)
        labels = np.arange(num_channels) + 1

    channels_per_row = 32
    num_rows = int(np.ceil(np.sqrt(num_channels / channels_per_row)))
    num_cols = int(np.floor(np.sqrt(num_channels / channels_per_row)))
    num_cols = max(1, num_cols)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 6 * num_rows))
    if num_rows == 1:
        axs.boxplot(tensor, labels=labels, **kwargs)
        axs.set_title(name)
        fig.subplots_adjust(hspace=0.4)
    else:
        axs = axs.ravel()
        axs[0].set_title(f"{name}")
        for i in range(num_rows * num_cols):
            start = i * channels_per_row
            end = (i + 1) * channels_per_row
            if start >= num_channels:
                break
            tensor_i = tensor[:, start: end]
            axs[i].boxplot(tensor_i, labels=labels[start: end], **kwargs)

    plt.show()
