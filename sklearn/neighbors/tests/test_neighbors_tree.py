# SPDX-License-Identifier: BSD-3-Clause

import itertools
import pickle

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
)

from sklearn.metrics import DistanceMetric
from sklearn.neighbors._ball_tree import (
    BallTree,
    kernel_norm,
)
from sklearn.neighbors._ball_tree import (
    NeighborsHeap64 as NeighborsHeapBT,
)
from sklearn.neighbors._ball_tree import (
    nodeheap_sort as nodeheap_sort_bt,
)
from sklearn.neighbors._ball_tree import (
    simultaneous_sort as simultaneous_sort_bt,
)
from sklearn.neighbors._kd_tree import (
    KDTree,
)
from sklearn.neighbors._kd_tree import (
    NeighborsHeap64 as NeighborsHeapKDT,
)
from sklearn.neighbors._kd_tree import (
    nodeheap_sort as nodeheap_sort_kdt,
)
from sklearn.neighbors._kd_tree import (
    simultaneous_sort as simultaneous_sort_kdt,
)
from sklearn.utils import check_random_state

rng = np.random.RandomState(42)
V_mahalanobis = rng.rand(3, 3)
V_mahalanobis = np.dot(V_mahalanobis, V_mahalanobis.T)

DIMENSION = 3

METRICS = {
    "euclidean": {},
    "manhattan": {},
    "minkowski": dict(p=3),
    "chebyshev": {},
    "seuclidean": dict(V=rng.random_sample(DIMENSION)),
    "mahalanobis": dict(V=V_mahalanobis),
}

KD_TREE_METRICS = ["euclidean", "manhattan", "chebyshev", "minkowski"]
BALL_TREE_METRICS = list(METRICS)


def dist_func(x1, x2, p):
    return np.sum((x1 - x2) ** p) ** (1.0 / p)


def compute_kernel_slow(Y, X, kernel, h):
    d = np.sqrt(((Y[:, None, :] - X) ** 2).sum(-1))
    norm = kernel_norm(h, X.shape[1], kernel)

    if kernel == "gaussian":
        return norm * np.exp(-0.5 * (d * d) / (h * h)).sum(-1)
    elif kernel == "tophat":
        return norm * (d < h).sum(-1)
    elif kernel == "epanechnikov":
        return norm * ((1.0 - (d * d) / (h * h)) * (d < h)).sum(-1)
    elif kernel == "exponential":
        return norm * (np.exp(-d / h)).sum(-1)
    elif kernel == "linear":
        return norm * ((1 - d / h) * (d < h)).sum(-1)
    elif kernel == "cosine":
        return norm * (np.cos(0.5 * np.pi * d / h) * (d < h)).sum(-1)
    else:
        raise ValueError("kernel not recognized")


def brute_force_neighbors(X, Y, k, metric, **kwargs):
    D = DistanceMetric.get_metric(metric, **kwargs).pairwise(Y, X)
    ind = np.argsort(D, axis=1)[:, :k]
    dist = D[np.arange(Y.shape[0])[:, None], ind]
    return dist, ind


@pytest.mark.parametrize("Cls", [KDTree, BallTree])
@pytest.mark.parametrize(
    "kernel", ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]
)
@pytest.mark.parametrize("h", [0.01, 0.1, 1])
@pytest.mark.parametrize("rtol", [0, 1e-5])
@pytest.mark.parametrize("atol", [1e-6, 1e-2])
@pytest.mark.parametrize("breadth_first", [True, False])
def test_kernel_density(
    Cls, kernel, h, rtol, atol, breadth_first, n_samples=100, n_features=3
):
    rng = check_random_state(1)
    X = rng.random_sample((n_samples, n_features))
    Y = rng.random_sample((n_samples, n_features))
    dens_true = compute_kernel_slow(Y, X, kernel, h)

    tree = Cls(X, leaf_size=10)
    dens = tree.kernel_density(
        Y, h, atol=atol, rtol=rtol, kernel=kernel, breadth_first=breadth_first
    )
    assert_allclose(dens, dens_true, atol=atol, rtol=max(rtol, 1e-7))


@pytest.mark.parametrize("Cls", [KDTree, BallTree])
def test_neighbor_tree_query_radius(Cls, n_samples=100, n_features=10):
    rng = check_random_state(0)
    X = 2 * rng.random_sample(size=(n_samples, n_features)) - 1
    query_pt = np.zeros(n_features, dtype=float)

    eps = 1e-15  # roundoff error can cause test to fail
    tree = Cls(X, leaf_size=5)
    rad = np.sqrt(((X - query_pt) ** 2).sum(1))

    for r in np.linspace(rad[0], rad[-1], 100):
        ind = tree.query_radius([query_pt], r + eps)[0]
        i = np.where(rad <= r + eps)[0]

        ind.sort()
        i.sort()

        assert_array_almost_equal(i, ind)


@pytest.mark.parametrize("Cls", [KDTree, BallTree])
def test_neighbor_tree_query_radius_distance(Cls, n_samples=100, n_features=10):
    rng = check_random_state(0)
    X = 2 * rng.random_sample(size=(n_samples, n_features)) - 1
    query_pt = np.zeros(n_features, dtype=float)

    eps = 1e-15  # roundoff error can cause test to fail
    tree = Cls(X, leaf_size=5)
    rad = np.sqrt(((X - query_pt) ** 2).sum(1))

    for r in np.linspace(rad[0], rad[-1], 100):
        ind, dist = tree.query_radius([query_pt], r + eps, return_distance=True)

        ind = ind[0]
        dist = dist[0]

        d = np.sqrt(((query_pt - X[ind]) ** 2).sum(1))

        assert_array_almost_equal(d, dist)


@pytest.mark.parametrize("Cls", [KDTree, BallTree])
@pytest.mark.parametrize("dualtree", (True, False))
def test_neighbor_tree_two_point(Cls, dualtree, n_samples=100, n_features=3):
    rng = check_random_state(0)
    X = rng.random_sample((n_samples, n_features))
    Y = rng.random_sample((n_samples, n_features))
    r = np.linspace(0, 1, 10)
    tree = Cls(X, leaf_size=10)

    D = DistanceMetric.get_metric("euclidean").pairwise(Y, X)
    counts_true = [(D <= ri).sum() for ri in r]

    counts = tree.two_point_correlation(Y, r=r, dualtree=dualtree)
    assert_array_almost_equal(counts, counts_true)


@pytest.mark.parametrize("NeighborsHeap", [NeighborsHeapBT, NeighborsHeapKDT])
def test_neighbors_heap(NeighborsHeap, n_pts=5, n_nbrs=10):
    heap = NeighborsHeap(n_pts, n_nbrs)
    rng = check_random_state(0)

    for row in range(n_pts):
        d_in = rng.random_sample(2 * n_nbrs).astype(np.float64, copy=False)
        i_in = np.arange(2 * n_nbrs, dtype=np.intp)
        for d, i in zip(d_in, i_in):
            heap.push(row, d, i)

        ind = np.argsort(d_in)
        d_in = d_in[ind]
        i_in = i_in[ind]

        d_heap, i_heap = heap.get_arrays(sort=True)

        assert_array_almost_equal(d_in[:n_nbrs], d_heap[row])
        assert_array_almost_equal(i_in[:n_nbrs], i_heap[row])


@pytest.mark.parametrize("nodeheap_sort", [nodeheap_sort_bt, nodeheap_sort_kdt])
def test_node_heap(nodeheap_sort, n_nodes=50):
    rng = check_random_state(0)
    vals = rng.random_sample(n_nodes).astype(np.float64, copy=False)

    i1 = np.argsort(vals)
    vals2, i2 = nodeheap_sort(vals)

    assert_array_almost_equal(i1, i2)
    assert_array_almost_equal(vals[i1], vals2)


@pytest.mark.parametrize(
    "simultaneous_sort", [simultaneous_sort_bt, simultaneous_sort_kdt]
)
def test_simultaneous_sort(simultaneous_sort, n_rows=10, n_pts=201):
    rng = check_random_state(0)
    dist = rng.random_sample((n_rows, n_pts)).astype(np.float64, copy=False)
    ind = (np.arange(n_pts) + np.zeros((n_rows, 1))).astype(np.intp, copy=False)

    dist2 = dist.copy()
    ind2 = ind.copy()

    # simultaneous sort rows using function
    simultaneous_sort(dist, ind)

    # simultaneous sort rows using numpy
    i = np.argsort(dist2, axis=1)
    row_ind = np.arange(n_rows)[:, None]
    dist2 = dist2[row_ind, i]
    ind2 = ind2[row_ind, i]

    assert_array_almost_equal(dist, dist2)
    assert_array_almost_equal(ind, ind2)


@pytest.mark.parametrize("Cls", [KDTree, BallTree])
def test_gaussian_kde(Cls, n_samples=1000):
    # Compare gaussian KDE results to scipy.stats.gaussian_kde
    from scipy.stats import gaussian_kde

    rng = check_random_state(0)
    x_in = rng.normal(0, 1, n_samples)
    x_out = np.linspace(-5, 5, 30)

    for h in [0.01, 0.1, 1]:
        tree = Cls(x_in[:, None])
        gkde = gaussian_kde(x_in, bw_method=h / np.std(x_in))

        dens_tree = tree.kernel_density(x_out[:, None], h) / n_samples
        dens_gkde = gkde.evaluate(x_out)

        assert_array_almost_equal(dens_tree, dens_gkde, decimal=3)


@pytest.mark.parametrize(
    "Cls, metric",
    itertools.chain(
        [(KDTree, metric) for metric in KD_TREE_METRICS],
        [(BallTree, metric) for metric in BALL_TREE_METRICS],
    ),
)
@pytest.mark.parametrize("k", (1, 3, 5))
@pytest.mark.parametrize("dualtree", (True, False))
@pytest.mark.parametrize("breadth_first", (True, False))
def test_nn_tree_query(Cls, metric, k, dualtree, breadth_first):
    rng = check_random_state(0)
    X = rng.random_sample((40, DIMENSION))
    Y = rng.random_sample((10, DIMENSION))

    kwargs = METRICS[metric]

    kdt = Cls(X, leaf_size=1, metric=metric, **kwargs)
    dist1, ind1 = kdt.query(Y, k, dualtree=dualtree, breadth_first=breadth_first)
    dist2, ind2 = brute_force_neighbors(X, Y, k, metric, **kwargs)

    # don't check indices here: if there are any duplicate distances,
    # the indices may not match.  Distances should not have this problem.
    assert_array_almost_equal(dist1, dist2)


@pytest.mark.parametrize(
    "Cls, metric",
    [(KDTree, "euclidean"), (BallTree, "euclidean"), (BallTree, dist_func)],
)
@pytest.mark.parametrize("protocol", (0, 1, 2))
def test_pickle(Cls, metric, protocol):
    rng = check_random_state(0)
    X = rng.random_sample((10, 3))

    if hasattr(metric, "__call__"):
        kwargs = {"p": 2}
    else:
        kwargs = {}

    tree1 = Cls(X, leaf_size=1, metric=metric, **kwargs)

    ind1, dist1 = tree1.query(X)

    s = pickle.dumps(tree1, protocol=protocol)
    tree2 = pickle.loads(s)

    ind2, dist2 = tree2.query(X)

    assert_array_almost_equal(ind1, ind2)
    assert_array_almost_equal(dist1, dist2)

    assert isinstance(tree2, Cls)


# ---------------------------------------------------------------------------
# Decomposable Bregman divergences (KDTree only)

BREGMAN_METRICS = ["kl", "dkl", "is", "dis"]


def bregman_reference(metric, Q, X):
    """Divergence from each query row of Q to each data row of X."""
    Q = Q[:, None, :]
    X = X[None, :, :]
    if metric == "kl":
        return np.sum(Q * np.log(Q / X) - Q + X, axis=-1)
    if metric == "dkl":
        return np.sum(X * np.log(X / Q) - X + Q, axis=-1)
    if metric == "is":
        r = Q / X
        return np.sum(r - np.log(r) - 1, axis=-1)
    if metric == "dis":
        r = X / Q
        return np.sum(r - np.log(r) - 1, axis=-1)
    raise ValueError(metric)


def _positive_data(seed=0, n_samples=80, n_queries=15, n_features=DIMENSION):
    rng = check_random_state(seed)
    X = rng.uniform(0.05, 1.0, (n_samples, n_features))
    Y = rng.uniform(0.05, 1.0, (n_queries, n_features))
    return X, Y


@pytest.mark.parametrize("metric", BREGMAN_METRICS)
@pytest.mark.parametrize("k", (1, 3, 5))
@pytest.mark.parametrize("dualtree", (True, False))
@pytest.mark.parametrize("breadth_first", (True, False))
@pytest.mark.parametrize("leaf_size", (1, 5))
def test_kd_tree_query_bregman(metric, k, dualtree, breadth_first, leaf_size):
    X, Y = _positive_data()

    tree = KDTree(X, leaf_size=leaf_size, metric=metric)
    dist1, ind1 = tree.query(Y, k, dualtree=dualtree, breadth_first=breadth_first)

    D = bregman_reference(metric, Y, X)
    ind2 = np.argsort(D, axis=1)[:, :k]
    dist2 = np.take_along_axis(D, ind2, axis=1)

    assert_allclose(dist1, dist2, rtol=1e-10)
    # continuous random data: no ties, so the indices must agree too
    assert_array_equal(ind1, ind2)


def test_kd_tree_query_bregman_measured_from_query():
    # the tree measures the divergence from the query to the training
    # points; the dual divergence reverses the direction, and the two
    # disagree because a divergence is asymmetric
    X, Y = _positive_data(seed=1)
    d_kl, i_kl = KDTree(X, metric="kl").query(Y, 3)
    d_dkl, i_dkl = KDTree(X, metric="dkl").query(Y, 3)

    D_kl = bregman_reference("kl", Y, X)
    D_dkl = bregman_reference("dkl", Y, X)
    assert_allclose(d_kl, np.take_along_axis(D_kl, i_kl, axis=1))
    assert_allclose(d_dkl, np.take_along_axis(D_dkl, i_dkl, axis=1))
    assert not np.allclose(d_kl, d_dkl)


@pytest.mark.parametrize("metric", BREGMAN_METRICS)
@pytest.mark.parametrize("dtype", (np.float64, np.float32))
def test_kd_tree_query_bregman_dtype(metric, dtype):
    X, Y = _positive_data(seed=2)
    X = X.astype(dtype)
    Y = Y.astype(dtype)
    tree = KDTree(X, leaf_size=3, metric=metric)
    dist, ind = tree.query(Y, 3)
    D = bregman_reference(metric, Y.astype(np.float64), X.astype(np.float64))
    ind_ref = np.argsort(D, axis=1)[:, :3]
    rtol = 1e-4 if dtype == np.float32 else 1e-10
    assert_allclose(dist, np.take_along_axis(D, ind_ref, axis=1), rtol=rtol)


@pytest.mark.parametrize("metric", BREGMAN_METRICS)
@pytest.mark.parametrize("leaf_size", (1, 4))
def test_kd_tree_query_radius_bregman(metric, leaf_size):
    X, Y = _positive_data(seed=3)
    D = bregman_reference(metric, Y, X)
    r = np.median(D)

    tree = KDTree(X, leaf_size=leaf_size, metric=metric)
    ind, dist = tree.query_radius(Y, r, return_distance=True)
    counts = tree.query_radius(Y, r, count_only=True)

    for i in range(Y.shape[0]):
        expected = np.flatnonzero(D[i] <= r)
        assert_array_equal(np.sort(ind[i]), expected)
        assert_allclose(np.sort(dist[i]), np.sort(D[i][expected]), rtol=1e-10)
        assert counts[i] == expected.size


@pytest.mark.parametrize("metric", BREGMAN_METRICS)
@pytest.mark.parametrize("dualtree", (True, False))
def test_kd_tree_two_point_bregman(metric, dualtree):
    X, Y = _positive_data(seed=4)
    D = bregman_reference(metric, Y, X)
    r = np.linspace(D.min(), D.max(), 10)

    tree = KDTree(X, leaf_size=2, metric=metric)
    counts = tree.two_point_correlation(Y, r, dualtree=dualtree)
    expected = [(D <= ri).sum() for ri in r]
    assert_array_equal(counts, expected)


@pytest.mark.parametrize("metric", BREGMAN_METRICS)
@pytest.mark.parametrize("protocol", (0, 1, 2))
def test_kd_tree_pickle_bregman(metric, protocol):
    X, Y = _positive_data(seed=5)
    tree1 = KDTree(X, leaf_size=2, metric=metric)
    dist1, ind1 = tree1.query(Y, 3)

    tree2 = pickle.loads(pickle.dumps(tree1, protocol=protocol))
    dist2, ind2 = tree2.query(Y, 3)

    assert_allclose(dist1, dist2)
    assert_array_equal(ind1, ind2)


@pytest.mark.parametrize("metric", BREGMAN_METRICS)
def test_kd_tree_bregman_rejects_invalid_data(metric):
    X, _ = _positive_data(seed=6)
    with pytest.raises(ValueError, match="non-negative|strictly positive"):
        KDTree(-X, metric=metric)


@pytest.mark.parametrize("metric", BREGMAN_METRICS)
def test_ball_tree_rejects_bregman(metric):
    X, _ = _positive_data(seed=7)
    with pytest.raises(ValueError, match="not valid for BallTree"):
        BallTree(X, metric=metric)


# ---------------------------------------------------------------------------
# (1 + eps)-approximate queries


@pytest.mark.parametrize(
    "metric", ["euclidean", "manhattan", "chebyshev", "minkowski", "kl", "is"]
)
@pytest.mark.parametrize("eps", (0.0, 0.25, 1.0, 4.0))
@pytest.mark.parametrize("dualtree", (True, False))
@pytest.mark.parametrize("breadth_first", (True, False))
def test_kd_tree_query_eps(metric, eps, dualtree, breadth_first):
    X, Y = _positive_data(seed=8, n_samples=200, n_queries=30)
    kwargs = METRICS.get(metric, {})
    k = 5

    tree = KDTree(X, leaf_size=2, metric=metric, **kwargs)
    dist_exact, _ = tree.query(Y, k)
    dist_eps, _ = tree.query(
        Y, k, dualtree=dualtree, breadth_first=breadth_first, eps=eps
    )

    # the j-th neighbor found is at most (1 + eps) times as far as the true
    # j-th nearest neighbor, and never closer
    assert np.all(dist_eps >= dist_exact - 1e-12)
    assert np.all(dist_eps <= (1 + eps) * dist_exact + 1e-12)
    if eps == 0:
        assert_allclose(dist_eps, dist_exact)


def test_kd_tree_query_eps_is_approximate():
    # a large eps prunes enough that some returned neighbors are genuinely
    # approximate, while still respecting the (1 + eps) bound
    X, Y = _positive_data(seed=9, n_samples=2000, n_queries=100, n_features=8)
    tree = KDTree(X, leaf_size=5)

    d0, _ = tree.query(Y, 5)
    d1, _ = tree.query(Y, 5, eps=2.0)
    assert np.all(d1 <= 3 * d0 + 1e-12)
    assert np.any(d1 > d0 + 1e-12)


def test_kd_tree_query_eps_validation():
    X, Y = _positive_data(seed=10)
    tree = KDTree(X)
    with pytest.raises(ValueError, match="eps must be non-negative"):
        tree.query(Y, 1, eps=-0.5)
