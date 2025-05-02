import numba
import numpy as np
from typing import Tuple

@numba.njit(numba.boolean[:](numba.int64[:], numba.int64[:]), parallel=True)
def isin(a, b):
    out = np.empty(a.shape[0], dtype=numba.boolean)
    b = set(b)
    for i in numba.prange(a.shape[0]):
        if a[i] in b:
            out[i] = True
        else:
            out[i] = False
    return out


@numba.njit(numba.int64[:](numba.float64[:]))
def detect_peaks(x: list[float]):
    # find indexes of all peaks
    x = np.asarray(x)
    if len(x) < 3:
        return np.empty(1, np.int64)
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    indl = np.asarray(indnan)

    if indl.size != 0:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf

    vil = np.zeros(dx.size + 1)
    vil[:-1] = dx[:]  # hacky solution because numba does not like hstack tuple arrays
    vix = np.zeros(dx.size + 1)
    vix[1:] = dx[:]

    ind = np.unique(np.where((vil > 0) & (vix <= 0))[0])

    # handle NaN's
    # NaN's and values close to NaN's cannot be peaks
    if ind.size and indl.size:
        outliers = np.unique(np.concatenate((indnan, indnan - 1, indnan + 1)))
        booloutliers = isin(ind, outliers)
        booloutliers = np.invert(booloutliers)
        ind = ind[booloutliers]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]

    # eliminate redundant values

    return np.unique(ind)


@numba.jit(numba.types.Tuple((numba.float64[:], numba.float64[:]))(numba.float64[:]), nopython=True)
def itd_baseline_extract(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(data, dtype=np.float64)
    rotation = np.zeros_like(x)

    alpha = 0.5

    # Assuming `detect_peaks` is a previously defined function
    idx_max = np.asarray(detect_peaks(x))
    idx_min = np.asarray(detect_peaks(-x))

    val_max = x[idx_max]  # Get peaks based on indexes
    val_min = -x[idx_min]

    num_extrema = len(val_max) + len(val_min)

    extremabuffersize = num_extrema + 2
    extrema_indices = np.zeros(extremabuffersize, dtype=np.int64)
    extrema_indices[1:-1] = np.sort(np.unique(np.hstack((idx_max, idx_min))))
    extrema_indices[-1] = len(x) - 1

    baseline_knots = np.zeros(len(extrema_indices), dtype=np.float64)
    baseline_knots[0] = np.mean(x[:2])
    baseline_knots[-1] = np.mean(x[-2:])

    # Compute baseline knots
    for k in range(1, len(extrema_indices) - 1):
        baseline_knots[k] = alpha * (x[extrema_indices[k - 1]] + \
                                     (extrema_indices[k] - extrema_indices[k - 1]) / \
                                     (extrema_indices[k + 1] - extrema_indices[k - 1]) * \
                                     (x[extrema_indices[k + 1]] - x[extrema_indices[k - 1]])) + \
                            alpha * x[extrema_indices[k]]

    baseline_new = np.zeros_like(x, dtype=np.float64)

    # Fill baseline between extrema
    for k in range(len(extrema_indices) - 1):
        baseline_new[extrema_indices[k]:extrema_indices[k + 1]] = baseline_knots[k] + \
                                                                  (baseline_knots[k + 1] - baseline_knots[k]) / \
                                                                  (extrema_indices[k + 1] - extrema_indices[k]) * \
                                                                  (np.arange(
                                                                      extrema_indices[k + 1] - extrema_indices[k]) - 0)

    # Calculate the rotation (signal - baseline)
    rotation[:] = np.subtract(x, baseline_new)

    return rotation[:], baseline_new[:]


@numba.jit(numba.float64[:, :](numba.float64[:]), nopython=True)
def itd(data: np.ndarray) -> np.ndarray:
    rotations = np.zeros((1, len(data)), dtype=np.float64)  # Only storing the first rotation
    baselines = np.zeros((1, len(data)), dtype=np.float64)
    rotation_ = np.zeros((len(data)), dtype=np.float64)
    baseline_ = np.zeros((len(data)), dtype=np.float64)

    # Perform the first baseline extraction
    rotation_[:], baseline_[:] = itd_baseline_extract(np.transpose(np.asarray(data, dtype=np.float64)))

    # Get peaks for the first baseline
    idx_max = np.asarray(detect_peaks(baseline_))
    idx_min = np.asarray(detect_peaks(-baseline_))
    num_extrema = len(idx_min) + len(idx_max)
    print(num_extrema)

    if num_extrema < 2:
        print("No more decompositions possible after first iteration.")
        return rotations  # Return empty or handle it as needed

    # Store the first rotation and baseline
    rotations[0, :] = rotation_[:]
    baselines[0, :] = baseline_[:]

    # Return only the first rotation
    return rotations
