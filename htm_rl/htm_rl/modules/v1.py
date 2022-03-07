import numpy as np
import matplotlib.pyplot as plt


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
) -> np.ndarray:
    """
    The Gabor filter: sin(2*pi*y/lambda) exp(-x^2/(2*sigma_x^2)-y^2/(2*sigma_y^2),
    where (x, y) are in basis rotated by theta.
    Parameters
    ----------
    size: int
        Determine the shape of the Gabor filter: ("size", "size").
    sigma_x: float
        Parameter for gaussian. The characteristic size
        of receptive field in x direction.
    sigma_y: float
        Parameter for gaussian. The characteristic size
        of receptive field in y direction.
    lambda_: float
        Parameter for sinus. The characteristic wavelength.
    theta: float
        The orientation of the filter (direction of the sinus wave).
    Returns
    -------
    filter_: np.ndarray
        Resulting rotated Gabor filter.
    """
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
    return filter_


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


def conv2d(signal_: np.ndarray, filter_: np.ndarray, stride: int = 1) -> np.ndarray:
    """
    The convolution of 2d signal and 2d filter.
    Parameters
    ----------
    signal_: np.ndarray
        Input 2 dimensional signal with the shape (H, W).
    filter_: np.ndarray
        2 dimensional filter for convolution with the shape (FH, FW).
    stride: int, default=1
        The size of the stride (the same in both directions).
    Returns
    -------
    convolution: np.ndarray
        The resulting convolution of signal and filter with the shape
        ((H - FH) // stride + 1, (W - FW) // stride + 1).
    """
    fh, fw = filter_.shape
    h, w = signal_.shape
    shape = (fh, fw, (h - fh) // stride + 1, (w - fw) // stride + 1)
    strides = signal_.strides + tuple(np.multiply(signal_.strides, stride))
    extended_matr = np.lib.stride_tricks.as_strided(signal_, shape=shape, strides=strides, writeable=False)
    convolution = np.einsum('ij,ijkl->kl', filter_, extended_matr)
    return convolution


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


def plot_3d_data(data: np.ndarray, columns: int):
    channels = data.shape[0]
    if channels % columns == 0:
        rows = channels // columns
    else:
        rows = channels // columns + 1
    fig, axs = plt.subplots(rows, columns, sharex=True, sharey=True)
    width = 20
    height = rows * data.shape[1] * width / (columns * data.shape[2])
    fig.set_size_inches(width, height)
    for i in range(rows):
        for j in range(columns):
            if j + i * columns < channels:
                axs[i, j].imshow(data[j + i * columns], cmap='gray', aspect='auto')
            else:
                axs[i, j].set_axis_off()
    plt.show()


def plot3d_3d_data(data: np.ndarray):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    f, i, j = data.nonzero()

    ax.scatter3D(f, i, j)
    ax.set_zlabel('Filters')
    plt.show()


class V1Simple:
    def __init__(self,
                 g_kernel_size: int,
                 g_stride: int,
                 g_sigma_x: float,
                 g_sigma_y: float,
                 g_lambda_: float,
                 g_filters: int,
                 activity_level: float,
                 g_pad: int = 0
                 ):
        self.gabor_filters = [
            create_gabor_filter(
                g_kernel_size, g_sigma_x, g_sigma_y, g_lambda_, 2 * np.pi * i / g_filters
            )
            for i in range(g_filters)
        ]
        self.ker_size = g_kernel_size
        self.pad = g_pad
        self.stride = g_stride
        self.activity_level = activity_level

    def output_shape(self, raw_image_shape: tuple[int, int]) -> tuple[int, int]:
        s1, s2 = raw_image_shape
        s1 = (s1 + 2 * self.pad - self.ker_size) // self.stride + 1
        s2 = (s2 + 2 * self.pad - self.ker_size) // self.stride + 1
        output = (s1, s2)
        return output

    def compute(self, image: np.ndarray) -> np.ndarray:
        gray = rgb2gray(image)
        gray = np.pad(gray, self.pad, 'constant', constant_values=0)
        preactivation = [
            conv2d(gray, filter_, self.stride)
            for filter_ in self.gabor_filters
        ]
        preactivation = np.array(preactivation)
        activation = kWTA(preactivation, self.activity_level)
        return activation


class V1Complex:

    def __init__(self,
                 g_kernel_size: int,
                 g_stride: int,
                 g_sigma: float,
                 input_shape: tuple[int, int],
                 activity_level: float,
                 ):
        self.activity_level = activity_level

        # gaus filters
        self.gaus_filter, wi, wj = create_gaus_filter(g_kernel_size, g_sigma)
        pad_w = (-np.min(wj), np.max(wj))
        pad_h = (-np.min(wi), np.max(wi))
        self.pad = (pad_h, pad_w)

        pad_shape = (
            input_shape[0] + pad_h[0] + pad_h[1],
            input_shape[1] + pad_w[0] + pad_w[1]
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
        self.output_shape = gaus_shape

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
            np.arange(4), (pad_shape[0] - 2, pad_shape[1] - 2, 2)
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
            np.arange(4), (pad_shape[0] - 2, pad_shape[1] - 2, 3, 2)
        ), (0, 1, 3, 2))
        self.off_mask = (off_kkk, off_iii, off_jjj)

    def gaus_max_pool(self, input: np.ndarray) -> np.ndarray:
        input_pad = np.pad(input, ((0, 0), *self.pad))
        preactivation = input_pad[:, self.gaus_mask[0], self.gaus_mask[1]] * self.gaus_filter
        preactivation = preactivation.max(axis=3)
        return preactivation

    def max_polarity(self, input: np.ndarray) -> np.ndarray:
        num_filteres = input.shape[0]
        angles = num_filteres // 2
        output = np.zeros((angles, input.shape[1], input.shape[2]))
        for i in range(angles):
            output[i] = input[[i, i + angles]].max(axis=0)
        return output

    def length_sum(self, input: np.ndarray) -> np.ndarray:
        preactivation = np.zeros_like(input)
        for i, filter_ in enumerate(self.length_filters):
            h, w = filter_.shape
            in_temp = np.pad(
                input[i], ((h // 2, h - h // 2 - 1), (w // 2, w - w // 2 - 1)),
                'constant', constant_values=0)
            preactivation[i] = conv2d(in_temp, filter_)
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


class V1:
    def __init__(self,
                 raw_image_shape: tuple[int, int],
                 complex_config: dict,
                 *simple_configs: dict,
                 ):
        self.num_paths = 0
        self.simple_cells = []
        self.complex_cells = []
        self.output_sizes = []
        for simple_config in simple_configs:
            self.num_paths += 1
            self.simple_cells.append(V1Simple(**simple_config))
            input_shape = self.simple_cells[-1].output_shape(raw_image_shape)
            self.complex_cells.append(
                V1Complex(**complex_config, input_shape=input_shape)
            )
            s1, s2 = self.complex_cells[-1].output_shape
            self.output_sizes.append(s1 * s2 * 20)

        self.output_sdr_size = sum(self.output_sizes)

    def compute(self, img: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        dense = []
        sparse = []
        for i in range(self.num_paths):
            sim = self.simple_cells[i].compute(img)
            com = self.complex_cells[i].compute(sim)
            dense.append(com)
            inds = np.nonzero(com.flatten())[0]
            sparse.append(inds)
        return sparse, dense
