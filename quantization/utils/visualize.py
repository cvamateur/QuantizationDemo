import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import quantization as q


def print_styles():
    from pprint import pprint
    pprint(matplotlib.style.available)


def plot_tensor_histogram(weights, name, bitwidth=32):
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


def plot_tensor_statistics(tensor, dim: int = None, style: str = "seaborn-v0_8-dark"):
    if isinstance(tensor, q.t_Tensor):
        tensor = tensor.detach().cpu().numpy()
    assert isinstance(tensor, np.ndarray), "error: input must be numpy.array"

    matplotlib.style.use(style)

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
    fig, axs = plt.subplots(num_rows, 1, figsize=(10, 6 * num_rows))
    if num_rows == 1:
        axs.boxplot(tensor, showfliers=False, labels=labels)
        axs.set_title("Tensor Stats")
        fig.subplots_adjust(hspace=0.4)
    else:
        axs = axs.ravel()
        for i in range(num_rows):
            start = i * channels_per_row
            end = (i + 1) * channels_per_row
            tensor_i = tensor[:, start: end]
            axs[i].boxplot(tensor_i, showfliers=False, labels=labels[start: end])
            axs[i].set_title(f"Tensor Stats - {i}")

    plt.show()