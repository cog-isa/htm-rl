import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x)


def bsu(x, k=10):
    return 1/(1 + np.exp(-x*k))


def heatmap(image, cmap=None):
    cm = plt.get_cmap(cmap)
    colored = cm(image)
    return Image.fromarray((colored[:, :, :3] * 255).astype(np.uint8))
