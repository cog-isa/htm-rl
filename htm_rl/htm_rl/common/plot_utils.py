import matplotlib.pyplot as plt
import numpy as np


def plot_grid_image(img: np.ndarray, uncenter = False, grid = False):
    plt.imshow(img)
    if uncenter:
        shape = img.shape
        xticks = np.arange(shape[1] + 1)
        yticks = np.arange(shape[0], -1, -1)
        plt.xticks(xticks - .5, xticks)
        plt.yticks(yticks - .5, yticks)
    ax = plt.gca()
    ax.xaxis.tick_top()
    if grid:
        plt.grid()
    plt.show()