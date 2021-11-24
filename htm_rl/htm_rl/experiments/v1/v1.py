import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Conv2d
import torch


def rgb2gray(rgb_image: np.ndarray) -> np.ndarray:
    if rgb_image.dtype.char in np.typecodes["AllInteger"]:
        rgb = rgb_image.astype(float) / 255
    else:
        rgb = rgb_image
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def create_gabor_filter(
        size: int, sigma_x: float, sigma_y: float,
        lambda_: float, theta: float
) -> torch.Tensor:
    filter_ = np.zeros((size, size))
    start = - (size - 1) / 2
    for i in range(size):
        for j in range(size):
            x_ = start + j
            y_ = - start - i

            x = x_ * np.cos(theta) - y_ * np.sin(theta)
            y = y_ * np.cos(theta) + x_ * np.sin(theta)
            r_2 = x ** 2 / (2 * sigma_x ** 2) + y ** 2 / (2 * sigma_y ** 2)
            filter_[i, j] = np.sin(2 * np.pi * y / lambda_) * np.exp(-r_2)
    return torch.tensor(filter_)


def create_gaus_filter(
        size: int, sigma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    filter_ = np.zeros((size, size))
    wi = np.zeros((size, size), dtype=np.int32)
    wj = np.zeros((size, size), dtype=np.int32)

    start = - (size - 1) / 2
    for i in range(size):
        for j in range(size):
            x = start + j
            y = - start - i

            r_2 = (x ** 2 + y ** 2) / (2 * sigma ** 2)
            filter_[i, j] = np.exp(-r_2)

            i_ = start + i
            j_ = start + j
            wi[i, j] = np.ceil(i_)
            wj[i, j] = np.ceil(j_)
    filter_ = filter_ / filter_.sum()
    return filter_.flatten(), wi, wj


def create_gabor_convolution(
        size: int, stride: int, pad: int, sigma_x: float,
        sigma_y: float, lambda_: float, num: int
) -> Conv2d:
    if stride == 1:
        convolution = Conv2d(1, num, size, stride=stride, bias=False, padding='same')
    else:
        convolution = Conv2d(1, num, size, stride=stride, bias=False, padding=pad)
    convolution.weight.requires_grad_(False)

    for i in range(num):
        convolution.weight.data[i, 0] = create_gabor_filter(
            size, sigma_x, sigma_y, lambda_, i * 2 * np.pi / num
        )
    return convolution


class V1Simple:

    def __init__(self,
                 g_kernel_size: int,
                 g_stride: int,
                 g_pad: int,
                 g_sigma_x: float,
                 g_sigma_y: float,
                 g_lambda_: float,
                 g_filters: int,
                 activity_level: float
                 ):
        self.convolution = create_gabor_convolution(
            g_kernel_size, g_stride, g_pad, g_sigma_x,
            g_sigma_y, g_lambda_, g_filters
        )
        self.num_filters = g_filters
        self.activity_level = activity_level

    @property
    def gabor_filters(self):
        return self.convolution.weight.squeeze().numpy()

    def plot_filters(self):
        cols = int(np.sqrt(self.num_filters)) + 1
        rows = self.num_filters // cols + 1
        fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True)
        fig.set_size_inches(10, 5)
        for i in range(rows):
            for j in range(cols):
                if j + i * cols < self.num_filters:
                    axs[i, j].imshow(self.gabor_filters[j + i * cols], cmap='gray')
                else:
                    axs[i, j].set_axis_off()
        plt.show()

    def plot_activation(self, activation: np.ndarray):
        cols = int(np.sqrt(self.num_filters)) + 1
        rows = self.num_filters // cols + 1
        fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True)
        fig.set_size_inches(20, 10)
        for i in range(rows):
            for j in range(cols):
                if j + i * cols < self.num_filters:
                    axs[i, j].imshow(activation[j + i * cols], cmap='gray',  aspect='auto')
                else:
                    axs[i, j].set_axis_off()
        plt.show()

    def compute(self, image: np.ndarray) -> np.ndarray:
        gray = rgb2gray(image)
        gray = torch.tensor(gray.reshape((1, 1, *gray.shape)),  dtype=torch.float32)
        preactivation = self.convolution(gray).squeeze().numpy()

        # 1st step kWTA
        active_cells = np.argmax(preactivation, axis=0)
        cells_value = np.max(preactivation, axis=0)

        # 2nd step kWTA
        k = int(cells_value.size * self.activity_level)
        active_hypcol = np.argpartition(-cells_value, k, axis=None)[:k]
        active_rows = active_hypcol // cells_value.shape[1]
        active_cols = active_hypcol % cells_value.shape[1]

        # final activation
        activation = np.zeros_like(preactivation)
        i = np.arange(activation.shape[1])
        j = np.arange(activation.shape[2])
        ii, jj = np.meshgrid(i, j, indexing='ij')
        active_cells = active_cells[active_rows, active_cols]
        ii = ii[active_rows, active_cols]
        jj = jj[active_rows, active_cols]
        activation[active_cells, ii, jj] = cells_value.flatten()[active_hypcol]
        return activation


class V1SimpleMax:

    def __init__(self,
                 g_kernel_size: int,
                 g_stride: int,
                 g_sigma: float,
                 input_shape: tuple[int, int, int],
                 ):
        self.gaus_filter, wi, wj = create_gaus_filter(g_kernel_size, g_sigma)
        pad_w = (-np.min(wj), np.max(wj))
        pad_h = (-np.min(wi), np.max(wi))
        self.pad = (pad_h, pad_w)

        pad_shape = (
            input_shape[1] + pad_h[0] + pad_h[1],
            input_shape[2] + pad_w[0] + pad_w[1]
        )
        i = np.arange(0, pad_shape[0], g_stride)
        j = np.arange(0, pad_shape[1], g_stride)
        ii, jj = np.meshgrid(i, j, indexing='ij')
        ii = ii[pad_h[0] // g_stride + 1:ii.shape[0] - pad_h[1] // g_stride,
             pad_w[0] // g_stride + 1:ii.shape[1] - pad_w[1] // g_stride]
        jj = jj[pad_h[0] // g_stride + 1:jj.shape[0] - pad_h[1] // g_stride,
             pad_w[0] // g_stride + 1:jj.shape[1] - pad_w[1] // g_stride]
        self.i = ii[:, :, np.newaxis] + wi.flatten()
        self.j = jj[:, :, np.newaxis] + wj.flatten()

    def compute(self, input: np.ndarray) -> np.ndarray:
        input_pad = np.pad(input, ((0, 0), *self.pad))
        output = input_pad[:, self.i, self.j] * self.gaus_filter
        output = output.max(axis=3)
        return output
