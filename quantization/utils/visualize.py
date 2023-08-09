import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ..q_types import t_Tensor, t_ndarray

def print_styles():
    from pprint import pprint
    pprint(matplotlib.style.available)


def plot_tensor_histogram(weights, name, qc):
    if qc.bitwidth <= 8:
        qmin, qmax = qc.q_min, qc.q_max
        bins = np.arange(qmin, qmax + 2)
        align = "left"
    else:
        bins = min(128, int(weights.numel() * 0.8))
        align = "mid"

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(weights.detach().cpu().view(-1), bins=bins, density=True,
            align=align, color='b', alpha=0.5, edgecolor='b' if qc.bitwidth <= 4 else None)
    ax.set_xlabel(name)
    ax.set_ylabel("density")
    fig.suptitle(f'Histogram of Weights (bitwidth={qc.bitwidth} bits)')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()


def plot_tensor_statistics(tensor,
                           name: str,
                           dim: int = None,
                           percentile: float = 99.,
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
    num_rows = int(np.ceil(num_channels / channels_per_row))
    fig, axs = plt.subplots(num_rows, 1, figsize=(10, 4 * num_rows))
    if num_rows == 1:
        axs.boxplot(tensor, labels=labels, **kwargs)
        axs.set_title(name)
        fig.subplots_adjust(hspace=0.4)
    else:
        axs = axs.ravel()
        axs[0].set_title(f"{name}")
        for i in range(num_rows):
            start = i * channels_per_row
            end = (i + 1) * channels_per_row
            tensor_i = tensor[:, start: end]
            axs[i].boxplot(tensor_i, labels=labels[start: end], **kwargs)

    plt.show()
