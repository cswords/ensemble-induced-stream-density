from ._naming import (
    IsolationModel,
    model_partitioning_map,
    EstimatorType,
    PartitioningMethod,
)
from ._data_dependent import (
    DEMassEstimator,  # Data-dependent density estimator using iTree
    MassEstimator,    # Data-dependent mass estimator (generic)
    IsolationTransformer, # Data-dependent kernel/similarity transformer
    IncrementalMassEstimator, # Data-dependent incremental mass estimator using iTree
)
from ._data_independent import (
    DataIndependentEstimator, # Generic data-independent estimator
    DataIndependentDensityEstimator, # Data-independent density estimator using iTree
)
from ._outlier_iforest import IsolationForestAnomalyDetector # Data-dependent anomaly detector (iForest specific)
from ._outlier_mass import IsolationBasedAnomalyDetector # Data-dependent anomaly detector (mass-based)


def init_estimator(
    estimator_type, isolation_model, psi, t=100, data_dependent=True, **kwargs
):
    """Initialize an isolation-based estimator.

    This factory function selects and instantiates the appropriate estimator
    based on the specified type, isolation model, and other parameters.

    Parameters
    ----------
    estimator_type : EstimatorType or str
        The type of estimator to initialize (e.g., DENSITY, MASS, ANOMALY, KERNEL).
        Can be an Enum member or its string value.
    isolation_model : IsolationModel or str
        The underlying isolation model to use (e.g., IFOREST, ANNE, INNE).
        Can be an Enum member or its string value.
    psi : int
        Subsample size (often referred to as `n_samples` in individual estimators).
        Used for models that involve subsampling, like those based on Isolation Trees.
    t : int, default=100
        Number of trees or base estimators, typically for ensemble models.
    data_dependent : bool, default=True
        If True, initializes a data-dependent estimator (learns from data structure).
        If False, initializes a data-independent estimator (uses predefined splits).
    **kwargs : dict
        Additional keyword arguments to pass to the constructor of the
        selected estimator class.

    Returns
    -------
    object
        An instance of the initialized estimator.

    Raises
    ------
    NotImplementedError
        If the combination of `estimator_type`, `isolation_model`,
        and `data_dependent` does not correspond to a defined estimator.
    """
    # Resolve IsolationModel if provided as a string
    resolved_isolation_model = (
        isolation_model
        if isinstance(isolation_model, IsolationModel)
        else IsolationModel(isolation_model)
    )
    # Determine the partitioning method associated with the isolation model
    partitioning_method = model_partitioning_map[resolved_isolation_model].value

    # Ensure estimator_type is the Enum member for consistent comparison
    if isinstance(estimator_type, str):
        estimator_type = EstimatorType(estimator_type)

    # --- Density Estimators ---
    if estimator_type == EstimatorType.DENSITY:
        if data_dependent:
            if partitioning_method == PartitioningMethod.ITREE:
                # Data-dependent density estimation using Isolation Trees (DEMass)
                return DEMassEstimator(psi, t, **kwargs)
            else:
                # Other data-dependent density estimators not implemented
                raise NotImplementedError(
                    f"Data-dependent DENSITY estimator for {isolation_model} "
                    f"(partitioning: {partitioning_method}) is not implemented."
                )
        else: # Data-independent
            if partitioning_method == PartitioningMethod.ITREE:
                # Data-independent density estimation using Isolation Trees
                return DataIndependentDensityEstimator(
                    psi, t, isolation_model=resolved_isolation_model, **kwargs
                )
            else:
                # Generic data-independent density estimator
                return DataIndependentEstimator(
                    psi, t, resolved_isolation_model, **kwargs
                )

    # --- Mass Estimators ---
    if estimator_type == EstimatorType.MASS:
        if data_dependent:
            if partitioning_method == PartitioningMethod.ITREE:
                # Data-dependent incremental mass estimation using Isolation Trees
                return IncrementalMassEstimator(psi, t, **kwargs)
            else:
                # Generic data-dependent mass estimator
                return MassEstimator(
                    psi, t, isolation_model=resolved_isolation_model, **kwargs
                )
        else: # Data-independent
            # Generic data-independent mass estimator
            return DataIndependentEstimator(
                psi, t, resolved_isolation_model, **kwargs # Note: DataIndependentEstimator is used for both mass and density here
            )

    # --- Anomaly Detectors ---
    if estimator_type == EstimatorType.ANOMALY:
        if data_dependent:
            if resolved_isolation_model == IsolationModel.IFOREST:
                # Specific anomaly detector for Isolation Forest
                return IsolationForestAnomalyDetector(psi, t, **kwargs)
            else:
                # Generic mass-based anomaly detector for other isolation models
                return IsolationBasedAnomalyDetector(
                    psi, t, isolation_model=resolved_isolation_model, **kwargs
                )
        else: # Data-independent
            # Data-independent anomaly detectors not implemented
            raise NotImplementedError(
                f"Data-independent ANOMALY estimator for {isolation_model} is not implemented."
            )

    # --- Kernel/Similarity Transformers ---
    if estimator_type == EstimatorType.KERNEL:
        if data_dependent:
            # Data-dependent kernel/similarity transformer
            return IsolationTransformer(
                psi, t, isolation_model=resolved_isolation_model, **kwargs
            )
        else: # Data-independent
            # Data-independent kernel/similarity transformers not implemented
            raise NotImplementedError(
                f"Data-independent KERNEL estimator for {isolation_model} is not implemented."
            )

    # Fallback if no valid estimator type matched
    raise NotImplementedError(
        f"Estimator type '{estimator_type}' with isolation model '{isolation_model}' "
        f"(data_dependent={data_dependent}) is not a valid combination."
    )
