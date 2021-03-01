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
