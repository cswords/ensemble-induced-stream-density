import pytest
import numpy as np
from Clustering._dbscan import DBSCAN, dbscan as dbscan_functional
from Common import ball_scale
from IsolationEstimators import IsolationModel
# It seems joblib.Parallel was used for running tests,
# but it might not be ideal for pytest structure.
# If specific tests require parallelism, they might need individual setup.
# For now, removing the direct use of joblib.Parallel from the test cases.

# Original xp = np, kept for consistency
xp = np # Note: _dbscan.py uses 'xp' for the array module, np here for tests.

# Test data from the original script
X_global = xp.array(
    [[-5.1, -5.1], [-4.6, -4.6], [-1.1, -1.1], [-0.6, -0.8], [6.6, 6.6]],
    dtype=np.float64,
)
Y_TRUE_GLOBAL = xp.array([1, 1, 2, 2, -1]) # Clusters are 1-indexed in original tests
X_BALL_GLOBAL = ball_scale(X_global)

# Parameters for tests
# Structure: test_name: (eps, minPts, psi, model_name_or_none, use_alpha, use_anomaly, expected_suffix)
PARAMS_DICT = {
    "l2_similarity": (1, 2, None, None, True, False, "_epsilon_neighbourhood"),
    "anne_similarity": (0.8, 2, 2, "anne", True, False, "_alpha_neighbourhood"),
    "soft_anne_similarity": (0.8, 2, 2, "soft_anne", True, False, "_alpha_neighbourhood"),
    "inne_similarity": (0.2, 2, 2, "inne", True, False, "_alpha_neighbourhood"),
    "isotropic_similarity": (0.1, 2, 2, "isotropic", True, False, "_alpha_neighbourhood"),
    "iforest_similarity": (0.8, 2, 2, "iforest", True, False, "_alpha_neighbourhood"),
    "anne_anomaly": (0.8, -3, 2, "anne", False, True, "_anomaly"),
    "soft_anne_anomaly": (0.8, -3, 2, "soft_anne", False, True, "_anomaly"),
    "inne_anomaly": (0.8, -0.4, 2, "inne", False, True, "_anomaly"),
    "iforest_anomaly": (1, 1.59, 3, "iforest", False, True, "_anomaly"),
    "anne_mass": (0.8, 2.5, 2, "anne", False, False, "_mass"),
    "soft_anne_mass": (0.8, 2.5, 2, "soft_anne", False, False, "_mass"),
    "inne_mass": (1, 1, 2, "inne", False, False, "_mass"),
    "isotropic_mass": (1, 2, 3, "isotropic", False, False, "_mass"),
    "iforest_mass": (1, 2.5, 2, "iforest", False, False, "_mass"),
}

@pytest.mark.parametrize(
    "test_name, params_tuple", list(PARAMS_DICT.items())
)
def test_dbscan_variants(test_name, params_tuple):
    """
    Tests DBSCAN with various configurations including different isolation models,
    similarity measures, anomaly scores, and mass-based approaches.
    """
    eps, minPts, psi, model_name_str, use_alpha, use_anomaly, _ = params_tuple

    actual_model_name = None if model_name_str == "l2_similarity" or model_name_str is None else model_name_str

    rotation = False
    X_to_use = X_BALL_GLOBAL if rotation and actual_model_name and "iforest" in actual_model_name else X_global

    kwargs_for_model = {}
    if actual_model_name:
        kwargs_for_model['rotation'] = rotation

    dbscan_model = DBSCAN(
        eps=eps,
        core_threshold=minPts,
        psi=psi,
        isolation_model_name=actual_model_name,
        use_alpha_neighbourhood_mass=use_alpha,
        use_anomaly_score=use_anomaly,
        **kwargs_for_model
    )

    labels = dbscan_model.fit_predict(X_to_use)

    np.testing.assert_array_equal(labels, Y_TRUE_GLOBAL,
                                   err_msg=f"Test failed for: {test_name}")

# New test functions
def test_dbscan_empty_input():
    """Test DBSCAN with empty input array."""
    X_empty = np.array([]).reshape(0, 2)
    # DBSCAN class fit method might raise error or return specific state
    # For _dbscan function, it expects n > 0.
    # Let's test the DBSCAN class interface.
    # The _dbscan core function would return (array([-1]), 0) for n=0 if it didn't error earlier.
    # However, m_sim.shape[0] would be 0.
    # If X is empty, X_ is set. Then m_sim computation might fail or result in 0x0.
    # If m_sim is 0x0, then n=0. `xp.zeros([n]) -1` would be `xp.zeros([0]) -1` which is `array([])`.
    # `n_components` would be 0.
    # So, expected labels is an empty array, and n_components = 0.

    # The class DBSCAN().fit(X_empty) will create self.X_ as an empty array.
    # self.m_sim will be computed. If X is (0,2), m_sim will be (0,0) for distance version.
    # Then predict will call _dbscan(m_sim=(0,0), ...)
    # In _dbscan: n = m_sim.shape[0] -> n = 0
    # `if xp.sum(l_core) <= 0:` -> l_core is likely empty or all false. sum is 0.
    # `return xp.zeros([n]) - 1, 0` -> `xp.zeros([0]) -1`, which is `array([])`

    model = DBSCAN(eps=0.5, core_threshold=2)
    labels = model.fit_predict(X_empty)
    np.testing.assert_array_equal(labels, np.array([]),
                                   err_msg="Empty input should result in empty labels.")

def test_dbscan_single_cluster():
    """Test DBSCAN with data that should form a single cluster."""
    X = np.array([[1, 1], [1.1, 1.1], [1.2, 1.2], [0.9, 0.9], [0.8, 0.8]])
    # eps=0.5 should be large enough to connect all, minPts=2
    model = DBSCAN(eps=0.5, core_threshold=2, isolation_model_name=None)
    labels = model.fit_predict(X)
    # Expect all points in one cluster (label 1, as it's 1-indexed in _dbscan)
    expected_labels = np.array([1, 1, 1, 1, 1])
    np.testing.assert_array_equal(labels, expected_labels,
                                   err_msg="Should form a single cluster.")

def test_dbscan_multiple_clusters():
    """Test DBSCAN with data that should form multiple distinct clusters."""
    X = np.array([
        [1, 1], [1.1, 1.1], [1.2, 1.2],  # Cluster 1
        [5, 5], [5.1, 5.1], [5.2, 5.2],  # Cluster 2
    ])
    # eps=0.5, minPts=2
    model = DBSCAN(eps=0.5, core_threshold=2, isolation_model_name=None)
    labels = model.fit_predict(X)
    # Expected: points in first group get label 1, second group get label 2 (or vice-versa)
    # The exact label values (1 or 2) depend on component discovery order,
    # but all points in a group should have the same label, and different groups different labels.
    # Let's check for two unique labels (excluding -1 noise)
    unique_labels = np.unique(labels[labels != -1])
    assert len(unique_labels) == 2, "Should find two clusters."
    # And ensure points in same original group have same label
    assert labels[0] == labels[1] == labels[2], "First group should be one cluster."
    assert labels[3] == labels[4] == labels[5], "Second group should be one cluster."
    assert labels[0] != labels[3], "The two groups should have different cluster labels."

def test_dbscan_with_noise():
    """Test DBSCAN with data including noise points."""
    X = np.array([
        [1, 1], [1.1, 1.1], [1.2, 1.2],  # Cluster 1
        [5, 5], [5.1, 5.1], [5.2, 5.2],  # Cluster 2
        [10, 10]                         # Noise point
    ])
    model = DBSCAN(eps=0.5, core_threshold=2, isolation_model_name=None)
    labels = model.fit_predict(X)
    # Check noise point
    assert labels[6] == -1, "Point (10,10) should be noise."
    # Check clusters (similar to above, but ensure noise point doesn't interfere)
    unique_labels = np.unique(labels[labels != -1])
    assert len(unique_labels) == 2, "Should find two clusters among non-noise points."
    assert labels[0] == labels[1] == labels[2] and labels[0] != -1, "First group should be a cluster."
    assert labels[3] == labels[4] == labels[5] and labels[3] != -1, "Second group should be a cluster."
    assert labels[0] != labels[3], "The two clusters should have different labels."

def test_dbscan_different_eps_values():
    """Test the effect of different eps values on clustering."""
    X = np.array([
        [1, 1], [1.5, 1.5],  # Point 0, 1
        [3, 3], [3.5, 3.5]   # Point 2, 3
    ]) # Distance between (1.5,1.5) and (3,3) is sqrt((1.5)^2 + (1.5)^2) = sqrt(4.5) approx 2.12

    # Case 1: eps is small, points form two clusters
    model_small_eps = DBSCAN(eps=0.6, core_threshold=2, isolation_model_name=None)
    labels_small_eps = model_small_eps.fit_predict(X)
    unique_small = np.unique(labels_small_eps[labels_small_eps != -1])
    assert len(unique_small) == 2, "Small eps should create two clusters."

    # Case 2: eps is large enough to connect all points into one cluster
    model_large_eps = DBSCAN(eps=2.2, core_threshold=2, isolation_model_name=None)
    labels_large_eps = model_large_eps.fit_predict(X)
    unique_large = np.unique(labels_large_eps[labels_large_eps != -1])
    assert len(unique_large) == 1, "Large eps should create one cluster."
    np.testing.assert_array_equal(labels_large_eps, np.array([1,1,1,1]), "All points should be in one cluster for large eps")


def test_dbscan_different_minpts_values():
    """Test the effect of different core_threshold (minPts) values."""
    X = np.array([
        [1,1], [1.1,1.1], [1.2,1.2], # Group A
        [5,5], [5.1,5.1]             # Group B (only 2 points)
    ])

    # Case 1: minPts = 2. Group A is a cluster, Group B is a cluster.
    model_minpts_2 = DBSCAN(eps=0.5, core_threshold=2, isolation_model_name=None)
    labels_minpts_2 = model_minpts_2.fit_predict(X)
    # Expected: [1,1,1, 2,2] (or labels swapped)
    assert labels_minpts_2[0] != -1 and labels_minpts_2[0] == labels_minpts_2[1] == labels_minpts_2[2]
    assert labels_minpts_2[3] != -1 and labels_minpts_2[3] == labels_minpts_2[4]
    assert labels_minpts_2[0] != labels_minpts_2[3]
    assert len(np.unique(labels_minpts_2[labels_minpts_2 != -1])) == 2

    # Case 2: minPts = 3. Group A is a cluster, Group B becomes noise (or part of A if eps is large, but here eps is small).
    model_minpts_3 = DBSCAN(eps=0.5, core_threshold=3, isolation_model_name=None)
    labels_minpts_3 = model_minpts_3.fit_predict(X)
    # Expected: [1,1,1, -1,-1]
    assert labels_minpts_3[0] != -1 and labels_minpts_3[0] == labels_minpts_3[1] == labels_minpts_3[2]
    assert labels_minpts_3[3] == -1
    assert labels_minpts_3[4] == -1
    assert len(np.unique(labels_minpts_3[labels_minpts_3 != -1])) == 1

# Using the functional interface for one test as an example
def test_dbscan_functional_interface_basic():
    """Test the functional dbscan interface with a basic case."""
    X = np.array([
        [1, 1], [1.1, 1.1], # Cluster 1
        [5, 5], [5.1, 5.1], # Cluster 2
        [10, 10]            # Noise
    ])
    # Calculate full distance matrix for functional interface
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(X))

    labels, n_components = dbscan_functional(dist_matrix, eps=0.5, minPts=2)

    assert n_components == 2
    assert labels[0] == labels[1] and labels[0] != -1
    assert labels[2] == labels[3] and labels[2] != -1
    assert labels[0] != labels[2] # Different clusters
    assert labels[4] == -1 # Noise

# It's good practice to ensure all global setup like np.set_printoptions
# is not needed for tests to pass, or is handled per-test / via fixtures if essential.
# np.set_printoptions(precision=4, suppress=True, linewidth=150) # From original, likely not needed for test logic
# np.set_printoptions(precision=2) # From original

# The commented out tensorflow part is omitted as it was not active.
# The `isolation_models = [None] + [e.value for e in IsolationModel]` line is not directly used
# by the parametrized tests as model names are strings from PARAMS_DICT.
