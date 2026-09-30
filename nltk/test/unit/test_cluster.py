"""
Unit tests for nltk.cluster: the vector space clusterers' normalisation and
SVD preprocessing, cluster assignment, and the input shapes that must fail
cleanly and quickly instead of looping. Nothing is mocked: every test runs
the real clusterers on real arrays.
"""

import random
import time

import pytest

from nltk.cluster import EMClusterer, GAAClusterer, KMeansClusterer, euclidean_distance

np = pytest.importorskip("numpy")

VECTORS = np.array(
    [[3.0, 1.0, 0.0], [4.0, 1.0, 0.0], [-3.0, -1.0, 0.0], [-4.0, -1.0, 0.0]]
)

#: A refusal must come back well inside this; a looping fit never comes back.
QUICK = 5.0


def _kmeans(normalise=False, svd_dimensions=None, **kwargs):
    return KMeansClusterer(
        2,
        euclidean_distance,
        normalise=normalise,
        svd_dimensions=svd_dimensions,
        rng=random.Random(0),
        **kwargs,
    )


def _em(normalise=False, svd_dimensions=None, width=3):
    means = np.zeros((2, svd_dimensions or width))
    means[:, 0] = [1.0, -1.0]
    return EMClusterer(means, normalise=normalise, svd_dimensions=svd_dimensions)


def _raises_quickly(exc, fn, match=None):
    started = time.perf_counter()
    with pytest.raises(exc, match=match):
        fn()
    assert time.perf_counter() - started < QUICK


@pytest.mark.parametrize("normalise", [False, True])
@pytest.mark.parametrize("svd_dimensions", [None, 1, 2])
@pytest.mark.parametrize("assign_clusters", [False, True])
@pytest.mark.parametrize("algorithm", ["kmeans", "em"])
def test_cluster_assignments_after_preprocessing(
    normalise, svd_dimensions, assign_clusters, algorithm
):
    vectors = VECTORS.copy()
    if algorithm == "kmeans":
        clusterer = _kmeans(normalise, svd_dimensions)
    else:
        clusterer = _em(normalise, svd_dimensions)

    assignments = clusterer.cluster(vectors, assign_clusters=assign_clusters)
    classifications = [clusterer.classify(vector) for vector in vectors]

    assert classifications[0] == classifications[1]
    assert classifications[2] == classifications[3]
    assert classifications[0] != classifications[2]
    if assign_clusters:
        assert assignments == classifications
    else:
        assert assignments is None
    np.testing.assert_array_equal(vectors, VECTORS)


@pytest.mark.parametrize("normalise", [False, True])
@pytest.mark.parametrize("svd_dimensions", [None, 1])
def test_gaa_cluster_assignments(normalise, svd_dimensions):
    vectors = np.array([[3.0, 1.0], [4.0, 1.0], [-3.0, -1.0], [-4.0, -1.0]])
    clusterer = GAAClusterer(2, normalise=normalise, svd_dimensions=svd_dimensions)

    assignments = clusterer.cluster(vectors, assign_clusters=True)

    assert assignments == [clusterer.classify(vector) for vector in vectors]
    assert assignments[0] == assignments[1]
    assert assignments[2] == assignments[3]
    assert assignments[0] != assignments[2]
    # the centroids live in the space classify() maps a query into
    width = svd_dimensions or vectors.shape[1]
    assert all(centroid.shape == (width,) for centroid in clusterer._centroids)


@pytest.mark.parametrize("svd_dimensions", [None, 2])
def test_fitting_is_unchanged_by_the_assignment_step(svd_dimensions):
    # the assignment step only reads the fit: the same seed must give the
    # same means whether or not labels are requested
    fitted, assigned = _kmeans(True, svd_dimensions), _kmeans(True, svd_dimensions)
    fitted.cluster(VECTORS.copy())
    assigned.cluster(VECTORS.copy(), assign_clusters=True)
    for a, b in zip(fitted.means(), assigned.means()):
        np.testing.assert_array_equal(a, b)
    fitted, assigned = _em(True, svd_dimensions), _em(True, svd_dimensions)
    fitted.cluster(VECTORS.copy())
    assigned.cluster(VECTORS.copy(), assign_clusters=True)
    np.testing.assert_array_equal(fitted._means, assigned._means)


def test_the_reported_shape_clusters_with_svd():
    # issue #2339: 1124 dimensions reduced by SVD, then labels for every vector
    vectors = np.random.default_rng(0).normal(size=(120, 300))
    clusterer = _kmeans(svd_dimensions=60, repeats=1)
    labels = clusterer.cluster(vectors, assign_clusters=True)
    assert len(labels) == 120
    assert clusterer._Tt.shape == (60, 300)
    assert labels == [clusterer.classify(vector) for vector in vectors]


def test_a_list_of_lists_is_assigned_like_an_array():
    clusterer = _kmeans(svd_dimensions=2)
    labels = clusterer.cluster(
        [list(vector) for vector in VECTORS], assign_clusters=True
    )
    assert labels == _kmeans(svd_dimensions=2).cluster(
        VECTORS.copy(), assign_clusters=True
    )


def test_a_reduced_vector_is_not_a_valid_classify_input():
    # what cluster() used to hand classify(): the projection applied twice
    clusterer = _kmeans(svd_dimensions=2)
    clusterer.cluster(VECTORS.copy())
    reduced = clusterer.vector(VECTORS[0])
    assert reduced.shape == (2,)
    _raises_quickly(ValueError, lambda: clusterer.classify(reduced))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("svd_dimensions", [None, 2])
@pytest.mark.parametrize("algorithm", ["kmeans", "em"])
def test_a_non_finite_component_is_refused_before_fitting(
    bad, svd_dimensions, algorithm
):
    # a NaN or infinite component made the k-means and EM loops compare
    # a distance or likelihood that is never below the threshold, so they
    # ran forever (an infinite component also hangs inside numpy's SVD)
    vectors = VECTORS.copy()
    vectors[0, 0] = bad
    clusterer = (
        _kmeans(False, svd_dimensions)
        if algorithm == "kmeans"
        else _em(False, svd_dimensions)
    )
    _raises_quickly(
        ValueError, lambda: clusterer.cluster(vectors, assign_clusters=True), "finite"
    )


def test_a_zero_vector_cannot_be_normalised():
    vectors = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    _raises_quickly(
        ValueError,
        lambda: _kmeans(True).cluster(vectors, assign_clusters=True),
        "zero vector",
    )
    clusterer = _kmeans(True)
    clusterer.cluster(vectors[1:])
    _raises_quickly(ValueError, lambda: clusterer.classify(vectors[0]), "zero vector")


@pytest.mark.parametrize("svd_dimensions", [0, 3, 10**9])
def test_svd_dimensions_at_or_past_the_width_skip_the_reduction(svd_dimensions):
    clusterer = _kmeans(svd_dimensions=svd_dimensions)
    labels = clusterer.cluster(VECTORS.copy(), assign_clusters=True)
    assert clusterer._Tt is None
    assert labels == [clusterer.classify(vector) for vector in VECTORS]


def test_svd_dimensions_out_of_range_fail_cleanly():
    _raises_quickly(
        ValueError, lambda: _kmeans(svd_dimensions=-1).cluster(VECTORS.copy())
    )
    _raises_quickly(
        TypeError, lambda: _kmeans(svd_dimensions=1.5).cluster(VECTORS.copy())
    )
    # more dimensions than samples but fewer than the width: numpy refuses the shapes
    two = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    _raises_quickly(ValueError, lambda: _kmeans(svd_dimensions=3).cluster(two))


def test_malformed_inputs_fail_cleanly():
    # pinned so each stays a quick refusal rather than a silent or looping fit
    _raises_quickly(AssertionError, lambda: _kmeans().cluster([]))
    _raises_quickly(TypeError, lambda: _kmeans().cluster(v for v in VECTORS))
    ragged = [np.array([1.0, 2.0]), np.array([1.0]), np.array([2.0, 3.0])]
    _raises_quickly(ValueError, lambda: _kmeans(svd_dimensions=1).cluster(ragged))
    _raises_quickly(ValueError, lambda: _kmeans().cluster(np.array([[1.0, 2.0]])))
    clusterer = _kmeans(svd_dimensions=2)
    clusterer.cluster(VECTORS.copy())
    _raises_quickly(ValueError, lambda: clusterer.classify(np.array([1.0, 2.0])))


def test_identical_vectors_empty_a_cluster_or_share_a_mean():
    same = np.array([[1.0, 2.0]] * 5)
    _raises_quickly(
        AssertionError, lambda: _kmeans().cluster(same, assign_clusters=True)
    )
    labels = _kmeans(avoid_empty_clusters=True).cluster(same, assign_clusters=True)
    assert len(set(labels)) == 1
    assert _em(width=2).cluster(same, assign_clusters=True) == [0] * 5
