import numpy as np
from Common import get_array_module
from scipy.sparse.csgraph import connected_components


def _adj(m_sim, eps, using_similarity=False):
    """Compute the adjacency matrix based on a similarity/distance threshold.

    Parameters
    ----------
    m_sim : array-like
        The similarity (or distance) matrix between samples.
    eps : float
        The threshold value. If `using_similarity` is True, pairs with
        similarity >= `eps` are considered neighbors. Otherwise, pairs with
        distance <= `eps` are considered neighbors.
    using_similarity : bool, default=False
        If True, `m_sim` is treated as a similarity matrix.
        If False, `m_sim` is treated as a distance matrix.

    Returns
    -------
    m_adj : array-like
        The boolean adjacency matrix where True indicates neighboring points.
    """
    if using_similarity:
        m_adj = m_sim >= eps
    else:
        m_adj = m_sim <= eps
    return m_adj


def _dens(m_adj, xp=np):
    """Calculate the density of each point based on its adjacency matrix.

    The density is defined as the number of neighbors for each point.

    Parameters
    ----------
    m_adj : array-like
        The boolean adjacency matrix.
    xp : module, default=numpy
        The array module to use (e.g., numpy or cupy).

    Returns
    -------
    densities : array-like
        An array containing the density (number of neighbors) for each point.
    """
    return xp.sum(m_adj, axis=1)


def epsilon_neighbourhood_density(m_sim, eps, using_similarity=False, xp=np):
    """Calculate the density of points within an epsilon-neighborhood.

    This function first computes an adjacency matrix based on the epsilon
    threshold and then calculates the density for each point.

    Parameters
    ----------
    m_sim : array-like
        The similarity (or distance) matrix between samples.
    eps : float
        The threshold value for defining the neighborhood.
    using_similarity : bool, default=False
        If True, `m_sim` is treated as a similarity matrix.
        If False, `m_sim` is treated as a distance matrix.
    xp : module, default=numpy
        The array module to use (e.g., numpy or cupy).

    Returns
    -------
    densities : array-like
        An array containing the density for each point.
    """
    m_adj = _adj(m_sim, eps, using_similarity)
    return _dens(m_adj, xp)


def _dbscan(m_sim, eps, threshold, l_dens=None, using_similarity=False):
    """Perform DBSCAN clustering.

    Core implementation of the DBSCAN algorithm. It identifies core points,
    builds clusters from them, and assigns other points to clusters or
    marks them as noise.

    Parameters
    ----------
    m_sim : array-like
        The similarity (or distance) matrix between samples. Shape (n_samples, n_samples).
    eps : float
        The maximum distance (or minimum similarity) between two samples for
        one to be considered as in the neighborhood of the other.
    threshold : int
        The number of samples in a neighborhood for a point to be considered
        as a core point. This includes the point itself.
    l_dens : array-like, optional
        Precomputed densities for each point. If None, densities are computed
        based on `m_sim` and `eps`. Shape (n_samples,).
    using_similarity : bool, default=False
        If True, `m_sim` is a similarity matrix (higher values are better).
        If False, `m_sim` is a distance matrix (lower values are better).

    Returns
    -------
    labels : array-like
        Cluster labels for each point in the dataset. Noise points are given
        the label -1. Shape (n_samples,).
    n_components : int
        The number of clusters found, excluding noise.
    """
    xp, xpUtils = get_array_module(m_sim)
    n = m_sim.shape[0]

    m_adj = _adj(m_sim, eps, using_similarity)
    if l_dens is None:
        l_dens = _dens(m_adj, xp)

    # print(m_sim)
    # print(l_dens)
    l_core = l_dens >= threshold

    # All points are noise if there is no core points.
    if xp.sum(l_core) <= 0:
        return xp.zeros([n]) - 1, 0

    # Label the core points by extracting the connected components of the \epsilon-neighbourhood graph.
    m_adj_core = m_adj[l_core, :][:, l_core]
    n_components, labels_core = connected_components(xpUtils.to_numpy(m_adj_core))
    labels_core = xp.array(labels_core)

    # Assign labels to all points using the label of their nearest core points.
    if using_similarity:
        l_assign_core = xp.argmax(m_sim[:, l_core], axis=1)
    else:
        l_assign_core = xp.argmin(m_sim[:, l_core], axis=1)
    labels = xp.take(labels_core, l_assign_core)

    # Label points as anomaly if their are outside of the \epsilon-neighbourhood of their nearest core points.
    if using_similarity:
        labels = xp.where(m_sim[xp.arange(n), l_assign_core] >= eps, labels + 1, -1)
    else:
        labels = xp.where(m_sim[xp.arange(n), l_assign_core] <= eps, labels + 1, -1)

    return labels, n_components


def dbscan(m_dis, eps, minPts):
    """Perform DBSCAN clustering using a distance matrix.

    User-friendly interface for DBSCAN clustering.

    Parameters
    ----------
    m_dis : array-like
        Distance matrix between samples. Shape (n_samples, n_samples).
    eps : float
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other.
    minPts : int
        The number of samples in a neighborhood for a point to be considered
        as a core point. This includes the point itself. Corresponds to
        `threshold` in `_dbscan`.

    Returns
    -------
    labels : array-like
        Cluster labels for each point in the dataset. Noise points are given
        the label -1.
    n_components : int
        The number of clusters found, excluding noise.
    """
    labels, n_components = _dbscan(
        m_dis, eps, minPts, l_dens=None, using_similarity=False
    )
    return labels, n_components


from sklearn.base import BaseEstimator, ClusterMixin
from IsolationEstimators import EstimatorType, IsolationModel, init_estimator


class DBSCAN(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        eps,
        core_threshold,
        psi,
        isolation_model_name=None,
        use_alpha_neighbourhood_mass=True,
        use_anomaly_score=False,
        dtype=np.float32,
        **kwargs
    ):
        """Initialize the DBSCAN estimator.

        Parameters
        ----------
        eps : float
            The maximum distance (or minimum similarity if using certain isolation models)
            between two samples for one to be considered as in the neighborhood of the other.
            This is used to determine the neighborhood of points.
        core_threshold : int
            The number of samples in a neighborhood for a point to be considered as a core point.
            This includes the point itself. This is equivalent to `minPts` in traditional DBSCAN.
        psi : int
            Sample size parameter, often used by the underlying isolation models.
            The exact meaning can vary depending on the `isolation_model_name`.
        isolation_model_name : str, optional
            The name of the isolation model to use for calculating similarity or density.
            If None, standard Euclidean distance is typically used.
            Examples: "anne", "iforest", "isotropic".
        use_alpha_neighbourhood_mass : bool, default=True
            If True and an `isolation_model_name` is provided, the model's
            similarity measure (alpha neighbourhood mass) is used.
            The `eps` parameter will then apply to this similarity.
        use_anomaly_score : bool, default=False
            If True and an `isolation_model_name` is provided (and
            `use_alpha_neighbourhood_mass` is False), the model's anomaly scores
            are used as densities.
        dtype : type, default=np.float32
            The data type for calculations.
        **kwargs : dict
            Additional keyword arguments passed to the underlying isolation model constructor.
        """
        self.eps = eps
        self.core_threshold = core_threshold
        self.isolation_model_name = isolation_model_name
        self.use_alpha_neighbourhood_mass = use_alpha_neighbourhood_mass
        self.use_anomaly_score = use_anomaly_score

        if isolation_model_name is None:
            self.isolation_model = None
        elif use_alpha_neighbourhood_mass:
            self.isolation_model = init_estimator(
                EstimatorType.KERNEL,
                IsolationModel(isolation_model_name),
                psi,
                **kwargs
            )
        elif use_anomaly_score:
            self.isolation_model = init_estimator(
                EstimatorType.ANOMALY,
                IsolationModel(isolation_model_name),
                psi,
                mass_based=False,
                **kwargs
            )
        else:
            self.isolation_model = init_estimator(
                EstimatorType.MASS, IsolationModel(isolation_model_name), psi, **kwargs
            )

        self.dtype = dtype
        self.X_ = None
        return

    def fit(self, X, y=None):
        """Fit the DBSCAN model to the dataset X.

        Computes or prepares the similarity/distance matrix and densities
        required for the DBSCAN algorithm. If an isolation model is used,
        it is fitted to X.

        Parameters
        ----------
        X : array-like
            Input data. Shape (n_samples, n_features).
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        xp, xpUtils = get_array_module(X)
        self.X_ = xpUtils.cast(X, dtype=self.dtype)

        # The similarity matrix and density are calculated for tunning purpose.
        if self.isolation_model is not None:
            self.isolation_model.fit(self.X_)

        using_similarity = (
            self.isolation_model is not None and self.use_alpha_neighbourhood_mass
        )
        if using_similarity:
            m_sim = self.isolation_model.transform(self.X_, return_similarity=True)
        else:
            m_sim = xpUtils.norm(
                xp.expand_dims(self.X_, axis=0) - xp.expand_dims(self.X_, axis=1),
                ord=2,
                axis=2,
            )

        if self.use_alpha_neighbourhood_mass:
            l_dens = None
        elif self.isolation_model is not None:
            if self.use_anomaly_score:
                l_dens = self.isolation_model.score_samples(self.X_)
            else:
                l_dens = self.isolation_model.score(self.X_)
        else:
            l_dens = None

        self.using_similarity = using_similarity
        self.m_sim = m_sim
        self.l_dens = l_dens
        return self

    def predict(self, X, y=None):
        """Perform clustering on X and return cluster labels.

        Predicts the cluster labels for the data points in X.
        Currently, this method expects X to be the same data that was used
        during fitting, or None. Predicting on new, unseen data is not
        supported by the current implementation.

        Parameters
        ----------
        X : array-like or None
            Input data. Shape (n_samples, n_features). If None, it uses the
            data passed during `fit`. If not None, it must be identical to
            the data used in `fit`.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        labels : array-like
            Cluster labels for each point in X. Noise points are given
            the label -1. Shape (n_samples,).

        Raises
        ------
        NotImplementedError
            If `X` is provided and is different from the data used in `fit`.
        """
        xp, _ = get_array_module(self.X_)
        if X is not None and xp.any(self.X_ != X):
            raise NotImplementedError(
                "Predicting on new data (X different from fit data) is not implemented."
            )

        using_similarity = (
            self.isolation_model is not None and self.use_alpha_neighbourhood_mass
        )
        y, _ = _dbscan(
            self.m_sim, self.eps, self.core_threshold, self.l_dens, using_similarity
        )
        return y

    def fit_predict(self, X, y=None):
        """Fit the model and return cluster labels for the same data.

        A convenience method that first calls `fit(X, y)` and then
        `predict(X=None, y=y)`.

        Parameters
        ----------
        X : array-like
            Input data. Shape (n_samples, n_features).
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        labels : array-like
            Cluster labels for each point in X. Noise points are given
            the label -1. Shape (n_samples,).
        """
        return self.fit(X, y).predict(X=None)
