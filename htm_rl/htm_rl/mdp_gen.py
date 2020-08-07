import numpy as np
import matplotlib.pyplot as plt

from htm_rl.common.utils import timed


def gen_mdp():
    seed = 1337
    np.random.seed(seed)

    mdps = 2
    density = .55
    n = 5
    T = 0
    for _seed in np.random.randint(2**15, size=mdps):
        _, t = gen_mdp_with_seed2(n, _seed, density)
        T += t
    print(T)


@timed
def gen_mdp_with_seed2(n, seed, density):
    np.random.seed(seed)
    required_cells = int(density * n**2)
    a = np.zeros((n, n), dtype=np.bool)
    dirs = [(1, 0), (0, -1), (-1, 0), (0, 1)]

    p = .1/np.sqrt(n)
    q_fw = 1 - 1./np.sqrt(n)
    s0 = np.random.randint(n**2)
    d = np.random.randint(4)
    i, j = divmod(s0, n)
    a[i][j] = True
    n_cells = 0
    s = 0
    while n_cells < required_cells:
        s += 1
        moved = False
        if np.random.rand() < q_fw:
            new_i = max(0, min(n-1, i + dirs[d][0]))
            new_j = max(0, min(n-1, j + dirs[d][1]))
            if not a[new_i][new_j]:
                i, j = new_i, new_j
                a[i, j] = True
                n_cells += 1
                moved = True

        if not moved:
            dif_d = int(np.sign(.5 - np.random.rand()))
            d = (d + dif_d + 4) % 4

        if np.random.rand() < p:
            b = np.zeros_like(a, dtype=np.float)
            b[1:] += a[1:] * (~a[:-1])
            b[:-1] += a[:-1] * (~a[1:])
            b[:, 1:] += a[:, 1:] * (~a[:, :-1])
            b[:, :-1] += a[:, :-1] * (~a[:, 1:])
            # plt.imshow(b), plt.show()
            b /= b.sum()
            visited = np.flatnonzero(b)
            s0 = np.random.choice(visited, p=b.ravel()[visited])
            d = np.random.randint(4)
            i, j = divmod(s0, n)

    # print(s / (n**2))
    plt.imshow(a)
    plt.show()


def gen_mdp_with_seed(n, seed, density):
    a = np.zeros((n, n))
    dirs = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    np.random.seed(seed)

    k, m = 2*n, n
    seeds = np.random.randint(2**15, size=k)
    s = 0
    for _seed in seeds:
        np.random.seed(_seed)

        s0 = np.random.randint(n**2)
        d = np.random.randint(4)
        i, j = divmod(s0, n)
        a[i][j] = 1
        step = 0
        while step < m:
            s += 1
            action = np.random.choice(3, p=[.6, .2, .2])
            if action == 0:
                new_i = max(0, min(n-1, i + dirs[d][0]))
                new_j = max(0, min(n-1, j + dirs[d][1]))
                if new_i != i or new_j != j:
                    i, j = new_i, new_j
                    a[i][j] += 1
                    step += 1
            else:
                dif_d = int(np.sign(1.5 - action))
                d = (d + dif_d + 4) % 4

    ranks = a.ravel().argsort().argsort().reshape((n, n))
    sparsity = 1 - density
    cutoff = int(sparsity * n**2)
    mask = ranks >= cutoff
    print(s / (n**2))

    plt.imshow(a)
    plt.show()
    plt.imshow(mask)
    plt.show()
