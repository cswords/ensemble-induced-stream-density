import numpy as np
from sklearn.base import DensityMixin
from joblib import delayed
from ._bagging import BaseAdaptiveBaggingEstimator
from ._isolation_tree import IsolationTree
from Common import get_array_module, ball_samples
from ._naming import IsolationModel, get_partitioning_initializer


def _single_fit(bagger, X):
    transformer = bagger.partitioning_initializer()
    n, dims = X.shape
    xp, xpUtils = get_array_module(X)
    samples = None
    if isinstance(transformer, IsolationTree):
        if transformer.rotation:
            samples = ball_samples(transformer.psi, dims, xp=xp, linalg=xpUtils)
        elif transformer.global_boundaries is not None:
            global_lower_boundary, global_upper_boundary = transformer.global_boundaries
            samples = xp.random.uniform(
                low=global_lower_boundary, high=global_upper_boundary, size=(n, dims)
            )
    if samples is None:
        samples = xp.random.uniform(
            low=X.min(axis=0), high=X.max(axis=0), size=(n, dims)
        )
    transformer.fit(samples)
    indices = transformer.transform(X)
    transformer.region_mass_ = xp.sum(
        xp.equal(indices, xp.expand_dims(xp.arange(transformer.psi), axis=0)),
        axis=0,
    )
    return transformer


def _single_partial_fit(transformer, X):
    xp, _ = get_array_module(X)
    indices = transformer.transform(X)
    transformer.region_mass_ = transformer.region_mass_ + xp.sum(
        xp.equal(indices, xp.expand_dims(xp.arange(transformer.psi), axis=0)),
        axis=0,
    )
    return transformer


def _single_score(isolation_model, transformer, X, return_demass):
    xp, xpUtils = get_array_module(X)
    region_mass = transformer.region_mass_
    indices = transformer.transform(X)
    if return_demass:
        if isolation_model == IsolationModel.IFOREST.value:
            region_volumes = transformer.node_volumes_[transformer.node_is_leaf_]
        else:
            # Never gonna happen
            region_volumes = transformer.region_volumes_
        region_demass = (
            xpUtils.cast(region_mass, dtype=np.dtype(float)) / region_volumes
        )
        return xp.take(region_demass, xp.squeeze(indices, axis=1))
    else:
        return xp.take(region_mass, xp.squeeze(indices, axis=1))


class DataIndependentEstimator(BaseAdaptiveBaggingEstimator, DensityMixin):
    def __init__(
        self,
        psi,
        t,
        isolation_model=IsolationModel.IFOREST.value,
        n_jobs=16,
        verbose=0,
        parallel=None,
        **kwargs
    ):
        partitioning_initializer = get_partitioning_initializer(
            isolation_model, psi, **kwargs
        )
        super().__init__(partitioning_initializer, t, n_jobs, verbose, parallel)
        self.psi = psi
        self.isolation_model = isolation_model

    def fit(self, X, y=None):
        xp, xpUtils = get_array_module(X)
        if xp.any(xpUtils.norm(X, axis=1) > 1):
            raise NotImplementedError("The data need to be ball_scale-ed")

        self.transformers_ = self.parallel()(
            delayed(_single_fit)(self, X) for _ in range(self.t)
        )

        self.fitted = X.shape[0]
        return self

    def partial_fit(self, X, y=None):
        xp, xpUtils = get_array_module(X)
        if xp.any(xpUtils.norm(X, axis=1) > 1):
            raise NotImplementedError("The data need to be ball_scale-ed")

        if self.fitted == 0:
            return self.fit(X, y)

        self.transformers_ = self.parallel()(
            delayed(_single_partial_fit)(i, X) for i in self.transformers_
        )

        self.fitted = self.fitted + X.shape[0]
        return self

    def score(self, X, return_demass=False):
        xp, xpUtils = get_array_module(X)
        if xp.any(xpUtils.norm(X, axis=1) > 1):
            raise NotImplementedError("The data need to be ball_scale-ed")

        if return_demass and self.isolation_model != IsolationModel.IFOREST.value:
            return NotImplementedError()

        all_results = self.parallel()(
            delayed(_single_score)(self.isolation_model, i, X, return_demass)
            for i in self.transformers_
        )
        return xp.average(xp.array(all_results), axis=0)


class DataIndependentDensityEstimator(DataIndependentEstimator):
    def __init__(
        self,
        psi,
        t,
        isolation_model=IsolationModel.IFOREST.value,
        n_jobs=16,
        verbose=0,
        parallel=None,
    ):
        if isolation_model != IsolationModel.IFOREST.value:
            raise NotImplementedError()
        super().__init__(psi, t, isolation_model, n_jobs, verbose, parallel)

    def score(self, X):
        return super().score(X, return_demass=True)
