import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def heatmap(image, cmap=None):
    cm = plt.get_cmap(cmap)
    colored = cm(image)
    return Image.fromarray((colored[:, :, :3] * 255).astype(np.uint8))
