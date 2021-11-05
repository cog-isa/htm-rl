from htm_rl.agents.cc.utils import ImageMovement
from htm_rl.agents.cc.spatial_pooler import TemporalDifferencePooler
from htm_rl.agents.cc.temporal_memory import GeneralFeedbackTM
from sklearn.datasets import load_digits
from sklearn.preprocessing import binarize
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def main(seed):
    X, y = load_digits(return_X_y=True)
    X = binarize(X, threshold=X.mean())
    print(X[0].reshape(8, 8), y[0])
    plt.imshow(X[0].reshape(8, 8))
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=seed
    )


if __name__ == '__main__':
    main(543)
