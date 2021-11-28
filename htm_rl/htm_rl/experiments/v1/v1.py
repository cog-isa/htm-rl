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


def conv2d(signal_: np.ndarray, filter_: np.ndarray) -> np.ndarray:
    h, w = filter_.shape
    shape = (h, w)
    signal_ = np.pad(signal_, ((h // 2, h - h // 2 - 1), (w // 2, w - w // 2 - 1)),
                     'constant', constant_values=0)
    s = shape + tuple(np.subtract(signal_.shape, shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    sub_matr = strd(signal_, shape=s, strides=signal_.strides * 2)
    return np.einsum('ij,ijkl->kl', filter_, sub_matr)


def relu(x):
    out = np.zeros_like(x)
    out[x > 0] = x[x > 0]
    return out


def nxx1(x, gamma):
    mask = x == 0
    out = np.zeros_like(x)
    out[~mask] = 1 / (1 + 1 / (gamma * x[~mask]))
    return out


def kWTA(preactivation: np.ndarray, activity_level: float) -> np.ndarray:
    preact = relu(preactivation)

    # 1st step kWTA
    cells_value = np.max(preact, axis=0)

    # 2nd step kWTA
    threshold = np.quantile(cells_value, 1 - activity_level)

    # final activation
    activation = nxx1(relu(preact - threshold), 1)
    return activation


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
        activation = kWTA(preactivation, self.activity_level)
        return activation


class V1Complex:

    def __init__(self,
        g_kernel_size: int,
        g_stride: int,
        g_sigma: float,
        input_shape: tuple[int, int, int],
        activity_level: float,
    ):
        self.activity_level = activity_level

        # gaus filters
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
        ii = ii[:, :, np.newaxis] + wi.flatten()
        jj = jj[:, :, np.newaxis] + wj.flatten()
        self.gaus_mask = (ii, jj)
        gaus_shape = (ii.shape[0], ii.shape[1])

        # len sum filters
        self.length_filters = [
            np.array([[1 / 3, 1 / 3, 1 / 3]]),
            np.array(
                [[1 / 3, 0, 0],
                 [0, 1 / 3, 0],
                 [0, 0, 1 / 3]]
            ),
            np.array([[1 / 3], [1 / 3], [1 / 3]]),
            np.array(
                [[0, 0, 1 / 3],
                 [0, 1 / 3, 0],
                 [1 / 3, 0, 0]]
            ),
        ]

        # end stop
        pad_shape = (
            gaus_shape[0] + 2,
            gaus_shape[1] + 2
        )
        i = np.arange(pad_shape[0])
        j = np.arange(pad_shape[1])
        ii, jj = np.meshgrid(i, j, indexing='ij')
        ii = ii[1:ii.shape[0] - 1, 1:ii.shape[1] - 1]
        jj = jj[1:jj.shape[0] - 1, 1:jj.shape[1] - 1]

        shift_i = np.array([0, 1, 1, 1, 0, -1, -1, -1])
        shift_j = np.array([1, 1, 0, -1, -1, -1, 0, 1])
        kkk = np.tile(
            np.arange(4), (pad_shape[0]-2, pad_shape[1]-2, 2)
        )
        iii = shift_i + ii[:, :, np.newaxis]
        jjj = shift_j + jj[:, :, np.newaxis]
        self.lsum_mask = (kkk, iii, jjj)

        off_i = np.array([
            [1, 0, -1],
            [0, -1, -1],
            [-1, -1, -1],
            [-1, -1, -0],
            [-1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 0],
        ])
        off_j = np.array([
            [-1, -1, -1],
            [-1, -1, 0],
            [-1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, -1],
            [0, -1, -1],
        ])
        off_iii = off_i + ii[:, :, np.newaxis, np.newaxis]
        off_jjj = off_j + jj[:, :, np.newaxis, np.newaxis]
        off_kkk = np.transpose(np.tile(
            np.arange(4), (pad_shape[0]-2, pad_shape[1]-2, 3, 2)
        ), (0, 1, 3, 2))
        self.off_mask = (off_kkk, off_iii, off_jjj)

    def gaus_max_pool(self,  input: np.ndarray) -> np.ndarray:
        input_pad = np.pad(input, ((0, 0), *self.pad))
        preactivation = input_pad[:, self.gaus_mask[0], self.gaus_mask[1]] * self.gaus_filter
        preactivation = preactivation.max(axis=3)
        return preactivation

    def max_polarity(self, input: np.ndarray) -> np.ndarray:
        num_filteres = input.shape[0]
        angles = num_filteres // 2
        output = np.zeros((angles, input.shape[1], input.shape[2]))
        for i in range(angles):
            output[i] = input[[i, i+angles]].max(axis=0)
        return output

    def length_sum(self, input: np.ndarray) -> np.ndarray:
        preactivation = np.zeros_like(input)
        for i, filter_ in enumerate(self.length_filters):
            preactivation[i] = conv2d(input[i], filter_)
        return preactivation

    def end_stop(self, input: np.ndarray, lsum: np.ndarray) -> np.ndarray:
        lsum_pad = np.pad(lsum, [[0], [1], [1]])
        lsum_masked = lsum_pad[
            self.lsum_mask[0],
            self.lsum_mask[1],
            self.lsum_mask[2]
        ]
        input_pad = np.pad(input, [[0], [1], [1]])
        off_masked = input_pad[
            self.off_mask[0],
            self.off_mask[1],
            self.off_mask[2]
        ]
        off_masked = off_masked.max(axis=3)
        preactivation = lsum_masked - off_masked
        preactivation = np.transpose(preactivation, (2, 0, 1))
        return preactivation


    def compute(self, input: np.ndarray) -> np.ndarray:
        v1simplemax = self.gaus_max_pool(input)
        angles_only_preact = self.max_polarity(v1simplemax)
        v1lensum = self.length_sum(angles_only_preact)
        v1estop = self.end_stop(angles_only_preact, v1lensum)
        preactivation = np.concatenate((v1simplemax, v1lensum, v1estop), axis=0)
        activation = kWTA(preactivation, self.activity_level)
        return activation
