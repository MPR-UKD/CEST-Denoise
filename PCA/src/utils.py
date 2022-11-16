import numpy as np
from itertools import product


def img_to_casorati_matrix(img: np.array, mask: np.array = None) -> np.array:
    n, m, _ = img.shape
    if mask is None:
        mask = np.ones((n, m))
    casorati_matrix = []
    for i1, i2 in product(range(n), range(m)):
        if mask[i2, i1] == 0:
            continue
        casorati_matrix.append(img[i2, i1, :])
    return np.array(casorati_matrix)
