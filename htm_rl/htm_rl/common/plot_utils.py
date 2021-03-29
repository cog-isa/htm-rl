import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_grid_images(images: List[np.ndarray], uncenter=False, grid=False):
    if not isinstance(images, List):
        images = [images]

    fig, axes = plt.subplots(1, len(images))
    if len(images) == 1:
        axes = [axes]

    for ax, img in zip(axes, images):
        _plot_grid_image(ax, img, uncenter, grid)
    plt.show()


def _plot_grid_image(ax, img: np.ndarray, uncenter=False, grid=False):
    ax.imshow(img)
    if uncenter:
        shape = img.shape
        x_ticks = np.arange(shape[1] + 1)
        y_ticks = np.arange(shape[0], -1, -1)
        ax.xticks(x_ticks - .5, x_ticks)
        ax.yticks(y_ticks - .5, y_ticks)

    ax.xaxis.tick_top()
    if grid:
        ax.grid()


def store_environment_map(ind, env_map, env_name, seed, test_dir):
    h, w = env_map.shape

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xticks(np.arange(-.5, w, 1))
    ax.set_yticks(np.arange(-.5, h, 1))
    ax.set_xticklabels(np.arange(w))
    ax.set_yticklabels(np.arange(h))
    ax.grid(color='grey', linestyle='-', linewidth=1)
    ax.set_title(f'{env_name}, seed={seed}')

    ax.imshow(env_map)
    save_path = os.path.join(test_dir, f'{env_name}_map_{ind}_{seed}.svg')
    fig.savefig(save_path, dpi=120)
