from pathlib import Path
from typing import Union, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.image import AxesImage
from numpy import ma

from htm_rl.common.utils import ensure_list, safe_ith
from htm_rl.envs.biogwlab.move_dynamics import DIRECTIONS_ORDER


def plot_grid_images(
        images: Union[np.ndarray, list[np.ndarray]],
        titles: Union[str, list[str]] = None,
        show: bool = True,
        save_path: Optional[Path] = None,
        with_value_text_flags: list[bool] = None,
        cols_per_row: int = 5
):
    images = ensure_list(images)
    titles = ensure_list(titles)
    n_images = len(images)

    n_rows = (n_images - 1) // cols_per_row + 1
    n_cols = min(n_images, cols_per_row)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 5 * n_rows)
    )
    if n_images == 1:
        axes = [axes]

    for i in range(n_images):
        ax = axes[i] if n_images <= cols_per_row else axes[i//cols_per_row][i%cols_per_row]
        img = images[i]
        title = safe_ith(titles, i)
        with_value_text = safe_ith(with_value_text_flags, i)
        _plot_grid_image(ax, img, title=title, with_value_text=with_value_text)

    fig.tight_layout()

    if show:
        plt.show()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)


def store_environment_map(
        map_ind: int, env_map: Union[np.ndarray, list[np.ndarray]],
        env_name: str, env_seed: int, test_dir: Path
):
    env_map = ensure_list(env_map)
    titles = [f'{env_name}, seed={env_seed}']
    if len(env_map) > 1:
        titles.append('agent observation')

    save_path = test_dir.joinpath(f'map_{env_name}_{map_ind}_{env_seed}.svg')
    plot_grid_images(env_map, titles, show=False, save_path=save_path)


def _plot_grid_image(
        ax, data: np.ndarray, title: Optional[str] = None,
        with_value_text: bool = False
):
    if title is not None:
        ax.set_title(title)
    ax.xaxis.tick_top()

    if data.ndim == 3 and data.shape[2] == 4:
        plot_triangled(data, ax)
        return

    h, w = data.shape[:2]
    # labels: major
    ax.set_xticks(np.arange(w + 1))
    ax.set_yticks(np.arange(h + 1))
    ax.set_xticklabels(np.arange(w+1))
    ax.set_yticklabels(np.arange(h+1))
    # grid: minor
    ax.set_xticks(np.arange(w + 1) - .5, minor=True)
    ax.set_yticks(np.arange(h + 1) - .5, minor=True)
    ax.grid(which="minor", color='grey', linestyle='-', linewidth=.5)

    threshold = .03 if data.dtype == np.float else 2
    im = ax.imshow(
        data,
        norm=mpl.colors.SymLogNorm(linthresh=threshold, base=10)
    )
    if with_value_text:
        valfmt = '{x:.1f}' if data.dtype == np.float else '{x}'
        annotate_heatmap(im, data=data, valfmt=valfmt)


def plot_triangled(data, ax):
    h, w = data.shape[:2]
    x = np.linspace(0, w, 2 * w + 1) - .5
    y = np.linspace(0, h, 2 * h + 1) - .5
    points = np.meshgrid(x, y)
    pxs, pys = points[0].ravel(), points[1].ravel()

    x_data, y_data = [], []
    for y_shit in range(h):
        for x_shift in range(w):
            up_x = 0, 1, 2
            up_y = 0, 1, 0

            down_x = 0, 1, 2
            down_y = 2, 1, 2

            left_x = 0, 1, 0
            left_y = 0, 1, 2

            right_x = 2, 1, 2
            right_y = 0, 1, 2

            assert DIRECTIONS_ORDER[0] == 'right'
            ixs = np.vstack((right_x, down_x, left_x, up_x)) + 2 * x_shift
            iys = np.vstack((right_y, down_y, left_y, up_y)) + 2 * y_shit
            x_data.append(ixs)
            y_data.append(iys)

    ixs = np.vstack(x_data)
    iys = np.vstack(y_data)
    ips = np.ravel_multi_index([iys, ixs], dims=points[0].shape)

    # labels: major
    ax.set_xticks(np.arange(w + 1))
    ax.set_yticks(np.arange(h + 1))
    ax.set_xticklabels(np.arange(w+1))
    ax.set_yticklabels(np.arange(h+1))
    # ax.grid(color='grey', linestyle='-', linewidth=2.)
    # grid: minor
    ax.set_xticks(np.arange(w + 1) - .5, minor=True)
    ax.set_yticks(np.arange(h + 1) - .5, minor=True)
    ax.grid(which="minor", color='grey', linestyle='-', linewidth=.5)
    ax.margins(x=0, y=0)
    ax.tripcolor(pxs, pys, ips, data.ravel())
    ax.invert_yaxis()


def annotate_heatmap(
        im: AxesImage, data: np.ndarray = None, valfmt="{x:.2f}",
        textcolors=("white", "black"), threshold=None, **textkw
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if isinstance(data, ma.MaskedArray) and data.mask[i, j]:
                continue
            over_threshold = im.norm(data[i, j]) > threshold
            kw.update(color=textcolors[int(over_threshold)])
            im.axes.text(j, i, valfmt(data[i, j], None), **kw)
