from pathlib import Path
from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np

from htm_rl.common.utils import ensure_list, safe_ith


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
        ax, img: np.ndarray, title: Optional[str] = None,
        with_value_text: bool = False
):
    h, w = img.shape[:2]
    ax.set_xticks(np.arange(-.5, w, 1))
    ax.set_yticks(np.arange(-.5, h, 1))
    ax.set_xticklabels(np.arange(w+1))
    ax.set_yticklabels(np.arange(h+1))
    ax.xaxis.tick_top()
    ax.grid(color='grey', linestyle='-', linewidth=.5)
    if title is not None:
        ax.set_title(title)
    if with_value_text:
        for i in range(h):
            for j in range(w):
                val = img[i, j]
                if img.dtype == np.float:
                    val = round(val, 1)
                ax.text(j, i, val, ha="center", va="center", color="w")
    ax.imshow(img)
