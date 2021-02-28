import matplotlib.pyplot as plt
import numpy as np


def plot_grid_image(img: np.ndarray, uncenter=False, grid=False):
    plt.imshow(img)
    if uncenter:
        shape = img.shape
        x_ticks = np.arange(shape[1] + 1)
        y_ticks = np.arange(shape[0], -1, -1)
        plt.xticks(x_ticks - .5, x_ticks)
        plt.yticks(y_ticks - .5, y_ticks)

    ax = plt.gca()
    ax.xaxis.tick_top()
    if grid:
        plt.grid()
    plt.show()
