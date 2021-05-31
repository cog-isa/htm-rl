import os
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_grid_images(images: List[np.ndarray]):
    if not isinstance(images, List):
        images = [images]

    fig, axes = plt.subplots(1, len(images))
    if len(images) == 1:
        axes = [axes]

    for ax, img in zip(axes, images):
        _plot_grid_image(ax, img)
    plt.show()


def store_environment_map(
        ind: int, env_map: Union[np.ndarray, List[np.ndarray]],
        env_name: str, seed: int, test_dir: Path
):
    fig: plt.Figure
    ax: plt.Axes

    if isinstance(env_map, list):
        env_map, observation = env_map
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    else:
        observation = None
        fig, axes = plt.subplots(1, 1, figsize=(6, 6))
        axes = [axes]

    # plot env map
    ax = axes[0]
    _plot_grid_image(ax, env_map)
    ax.set_title(f'{env_name}, seed={seed}')

    if observation is not None:
        ax = axes[1]
        _plot_grid_image(ax, observation)
        ax.set_title('agent observation')

    save_path = test_dir.joinpath(f'{env_name}_map_{ind}_{seed}.svg')
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def store_heatmap(ind, heatmap, name_str, test_dir: Path):
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    _plot_grid_image(ax, heatmap)

    ax.set_title(name_str)
    save_path = test_dir.joinpath(f'heatmap_{name_str}_{ind}.svg')
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _plot_grid_image(ax, img: np.ndarray):
    h, w = img.shape[:2]
    ax.set_xticks(np.arange(-.5, w, 1))
    ax.set_yticks(np.arange(-.5, h, 1))
    ax.set_xticklabels(np.arange(w+1))
    ax.set_yticklabels(np.arange(h+1))
    ax.xaxis.tick_top()
    ax.grid(color='grey', linestyle='-', linewidth=.5)
    ax.imshow(img)
