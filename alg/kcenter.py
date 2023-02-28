import numpy as np
from tqdm import tqdm


class KCenter:
    def __init__(self, K):
        self.K = K

    def fit_transform(self, x, verbose=False):
        # fit and return centers
        idx1 = np.random.choice(x.shape[0], 1)
        center = x[idx1]  # (1, d)
        idx = [idx1.item()]
        min_distances_to_centers = np.full([1, x.shape[0]], np.inf)
        for k in (tqdm if verbose else iter)(range(1, self.K)):
            distances = ((x[None, :, :] - center[:, None, :])
                         ** 2).sum(-1)  # 1, N
            min_distances_to_centers = np.vstack(
                [min_distances_to_centers, distances]).min(0)
            new_center_idx = np.argmax(min_distances_to_centers).item()
            idx.append(new_center_idx)
            center = x[[new_center_idx]]
        assert len(idx) == self.K
        return idx
