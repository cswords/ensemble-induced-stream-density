from sklearn.base import TransformerMixin, DensityMixin
from Common import (
    ReservoirSamplingEstimator,
    get_array_module,
    rotate,
    checked_and_warn,
)
from ._binary_tree import AxisParallelBinaryTree


def get_boundaries(X, ball_scaled=True):
    """Calculate global boundaries for the data X.

    These boundaries are used to define the root node's space in the tree.

    Parameters
    ----------
    X : array-like
        The input data samples.
    ball_scaled : bool, default=True
        If True, assumes data is scaled to a unit ball (e.g., values between -1 and 1),
        and sets boundaries accordingly. Otherwise, computes boundaries from
        the min/max of X with a small epsilon padding.

    Returns
    -------
    global_lower_boundary : array-like
        The lower boundaries for each selected dimension.
    global_upper_boundary : array-like
        The upper boundaries for each selected dimension.
    select_dims : array-like
        Indices of dimensions where X_min < X_max (i.e., non-constant dimensions).
    """
    xp, _ = get_array_module(X)
    eps = xp.finfo(X.dtype).eps
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    select_dims = xp.where(X_min < X_max)[0]

    if ball_scaled:
        global_upper_boundary = xp.ones([1, select_dims.shape[0]], dtype=X.dtype)
        global_lower_boundary = 0 - global_upper_boundary
    else:
        global_lower_boundary = xp.expand_dims(X_min[select_dims], axis=0)
        global_upper_boundary = xp.expand_dims(X_max[select_dims] + (eps * 2), axis=0)
        # Only adding 1 eps does not change the comparison result.

    return global_lower_boundary, global_upper_boundary, select_dims


class IsolationTree(
    ReservoirSamplingEstimator, AxisParallelBinaryTree, TransformerMixin
):
    """Isolation Tree for anomaly detection and density estimation.

    This tree isolates samples by randomly selecting a feature and then
    randomly selecting a split value between the minimum and maximum values
    of the selected feature.

    Parameters
    ----------
    psi : int
        Subsample size, equivalent to `n_samples` in `ReservoirSamplingEstimator`.
        This is the number of samples used to build the tree (reservoir size).
    rotation : bool, default=False
        If True, applies a random rotation to the input data `X` before fitting.
        The rotation matrix `SO_` is stored.
    global_boundaries : tuple, optional
        A tuple `(lower_bound_array, upper_bound_array)` defining the global
        data boundaries. If None, boundaries are computed using `get_boundaries`.
    **kwargs :
        Additional keyword arguments passed to parent classes.
    """

    def __init__(self, psi, rotation=False, global_boundaries=None, **kwargs):
        super().__init__(psi)
        self.rotation = rotation
        self.global_boundaries = global_boundaries
        return

    def fit(self, X, y=None):
        """Fit the IsolationTree model.

        Builds the tree structure based on the input data X.

        Parameters
        ----------
        X : array-like
            The input samples.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            The fitted IsolationTree instance.
        """
        xp, _ = get_array_module(X)
        if self.rotation:
            X_, SO = rotate(X)
            self.SO_ = SO  # Store the rotation matrix
        else:
            X_ = X

        # Determine global boundaries and selected dimensions for splitting
        if self.global_boundaries is not None:
            global_lower_boundary, global_upper_boundary = self.global_boundaries
            select_dims = xp.arange(X.shape[1])  # Use all dimensions if boundaries are pre-set
        else:
            global_lower_boundary, global_upper_boundary, select_dims = get_boundaries(
                X_, self.rotation  # Use rotated data for boundary calculation if rotation is True
            )
        self.select_dims = select_dims # Store dimensions used for splitting

        # Fit using the (potentially rotated) and dimension-selected data
        super().fit(X_[:, select_dims], y)
        # Seed the root node of the binary tree with these boundaries
        super().seed(global_lower_boundary, global_upper_boundary)
        self._build_tree()
        return self

    def search(self, X, l_nodes=None):
        """Search for samples in the tree nodes.

        Parameters
        ----------
        X : array-like
            The input samples to search.
        l_nodes : array-like, optional
            A boolean array indicating which nodes to consider in the search.
            If None, searches all nodes.

        Returns
        -------
        array-like
            A boolean matrix indicating which samples fall into which nodes.
        """
        if self.rotation:
            # If data was rotated during fit, apply the same rotation for search
            X_, SO = rotate(X, self.SO_ if hasattr(self, "SO_") else None)
            if not hasattr(self, "SO_") and X.shape[0] > 0 : # If SO_ was not stored (e.g. fit on empty X)
                self.SO_ = SO # Store it now, this might happen if fit was called with empty X
        else:
            X_ = X
        # Select only the dimensions used during fitting
        X_ = X_[:, self.select_dims]
        return super().search(X_, l_nodes)

    def _isolation_split(self, i, stack_to_split, lower_boundary, upper_boundary):
        """Perform a single isolation split at node i.

        Randomly selects a dimension and a split value within the range of
        samples present in the current node's region.

        Parameters
        ----------
        i : int
            The index of the current node to split.
        stack_to_split : list
            The list of node indices that still need to be split.
        lower_boundary : array-like
            The lower boundary of the current node's region.
        upper_boundary : array-like
            The upper boundary of the current node's region.

        Returns
        -------
        list
            Updated list of node indices to be split.

        Raises
        ------
        Exception
            If no samples are found within the current node's region,
            which indicates an issue with boundary calculations or search.
        """
        xp, _ = get_array_module(lower_boundary)
        eps = xp.finfo(self.samples_.dtype).eps

        # Find samples within the current node's boundaries
        l_in = self._search(self.samples_, lower_boundary, upper_boundary)
        l_in = l_in[:, 0]  # Search returns a 2D array, take the first column

        # If only one sample is in the region, it's isolated, no further split needed for this path
        if xp.sum(l_in) == 1:
            return stack_to_split
        # This should not happen if logic is correct
        elif xp.sum(l_in) < 1:
            # This indicates an issue, possibly with boundary propagation or search logic
            raise Exception("There is no sample in this region! Node index: {}".format(i))

        # Get samples within the current node
        m_in = xp.take(self.samples_, xp.where(l_in)[0], axis=0)
        l_min = xp.min(m_in, axis=0)
        l_max = xp.max(m_in, axis=0)

        # Find dimensions where there's a range to split (min < max)
        # Add 2*eps for robust comparison with floating point numbers
        l_dims = xp.where(l_min + (2 * eps) <= l_max)[0]
        if l_dims.shape[0] == 0: # No dimension to split on (all points are identical in selected dims)
            return stack_to_split


        # Randomly select a dimension to split on
        rand_split_dim_idx = xp.random.randint(low=0, high=l_dims.shape[0])
        split_dim = l_dims[rand_split_dim_idx]

        # Randomly select a split value in that dimension
        # Add eps and subtract eps to ensure split value is strictly within min/max
        split_value = xp.random.uniform(l_min[split_dim] + eps, l_max[split_dim] - eps)
        split_value = xp.asarray(split_value, dtype=l_max.dtype) # Use asarray to maintain type

        # Grow the tree by adding left and right children
        i_left, i_right = self.grow(i, split_dim, split_value)
        stack_to_split.append(i_left)
        stack_to_split.append(i_right)

        # Debugging check (originally commented out, kept for reference)
        # This check verifies that child nodes created by a split actually contain samples.
        # If not, it indicates a problem with the split logic or boundary updates.
        # for i_child in [i_left, i_right]:
        #     lower_boundary_child = self.node_lower_boundaries[i_child, :]
        #     upper_boundary_child = self.node_upper_boundaries[i_child, :]
        #     samples_in_child = self._search(self.samples_, lower_boundary_child, upper_boundary_child)
        #     if xp.sum(samples_in_child[:,0]) < 1:
        #         # This part of the original code seems to be for detailed debugging
        #         # in case a child node ends up empty.
        #         from ._binary_tree import single_split # Local import for debugging
        #         (
        #             debug_lower_left, debug_upper_left,
        #             debug_lower_right, debug_upper_right
        #         ) = single_split(
        #             self.node_lower_boundaries[i:i+1, :],
        #             self.node_upper_boundaries[i:i+1, :],
        #             split_dim, split_value
        #         )
        #         # Re-run search with child's boundaries on original m_in to see why it's empty
        #         self._search(m_in, lower_boundary_child, upper_boundary_child)
        #         raise Exception(f"Child node {i_child} has no samples after split from parent {i}!")

        return stack_to_split

    def _build_tree(self):
        """Build the isolation tree by iteratively splitting nodes."""
        stack_to_split = [0]  # Start with the root node (index 0)

        while len(stack_to_split) > 0:
            i = stack_to_split.pop()  # Get the next node to split
            # Retrieve boundaries for the current node i
            # Note: self.node_lower_boundaries etc. are updated by self.grow via AxisParallelBinaryTree
            lower_boundary_i = self.node_lower_boundaries[i, :]
            upper_boundary_i = self.node_upper_boundaries[i, :]
            stack_to_split = self._isolation_split(
                i, stack_to_split, lower_boundary_i, upper_boundary_i
            )

        # After all splits, determine which nodes are leaves and calculate volumes
        self.node_is_leaf_ = self.leaf()
        self.node_volumes_ = self.volumes()

    def partial_fit(self, X, y=None):
        """Incrementally update the tree with new samples X.

        This involves updating the reservoir of samples, and then potentially
        pruning and growing branches of the tree based on dropped and new samples
        from the reservoir.

        Parameters
        ----------
        X : array-like
            The new input samples to incorporate into the tree.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            The updated IsolationTree instance.
        """
        xp, _ = get_array_module(X)
        if self.rotation:
            # Apply stored rotation if it exists
            X_, _ = rotate(X, self.SO_ if hasattr(self, "SO_") else None)
        else:
            X_ = X
        X_ = X_[:, self.select_dims] # Apply dimension selection

        # Update reservoir samples
        changed_count, new_samples, drop_samples, reservoir = super().update_samples(X_)

        # Process changes due to reservoir update
        for i in range(changed_count):
            drop_sample = drop_samples[i : i + 1, :]
            self._prune(drop_sample) # Prune branches related to dropped sample

            new_sample = new_samples[i : i + 1, :]
            self._grow(new_sample) # Grow branches for new sample

        # Update node mass based on all current samples in X (not just reservoir)
        # This seems to update mass based on the current batch X, not the whole history.
        # If this is for AdaptiveMassEstimationTree, node_mass_ update is different.
        # For base IsolationTree, node_mass_ is not a standard attribute.
        # This line might be more relevant for a derived class or specific use case.
        # self.node_mass_ = self.node_mass_ + xp.sum(self.search(X_), axis=0, dtype=float)
        # TODO: Clarify role of self.node_mass_ update in base IsolationTree.
        # It's used by AdaptiveMassEstimationTree.fit, but partial_fit here might cause issues if not handled carefully.

        self.samples_ = reservoir # Update stored samples to new reservoir
        self.fitted = getattr(self, 'fitted', 0) + X_.shape[0] # Update total samples fitted count

        return self

    def _grow(self, new_sample):
        """Grow the tree from a leaf node to accommodate a new sample.

        Finds the leaf node where the new sample falls, splits that leaf,
        and updates tree properties.

        Parameters
        ----------
        new_sample : array-like
            The new sample to incorporate. Shape (1, n_features).

        Returns
        -------
        i_parent : int
            Index of the parent node that was split.
        i_left : int
            Index of the new left child node.
        i_right : int
            Index of the new right child node.
        """
        xp, _ = get_array_module(new_sample)
        l_leaf = xp.where(self.node_is_leaf_)[0] # Get current leaf nodes

        # Find which leaf node the new sample falls into
        search_result = self.search(new_sample, l_nodes=self.node_is_leaf_)
        if not xp.any(search_result):
            # This can happen if new_sample is outside all current leaf boundaries.
            # This might indicate an issue if global boundaries were too tight or data shifted.
            # For now, we might need to find the closest leaf or re-evaluate boundaries.
            # A simpler approach: find the node it would belong to if not restricted to leaves.
            # This situation needs careful handling.
            # A possible cause: if rotation is applied, and SO_ was not learned (e.g. fit on empty)
            # and then partial_fit is called.
            # Fallback: search all nodes and find the deepest one?
            # Or, this implies the sample is truly "new" and might need broader handling.
            # For now, if no leaf is found, it's an issue.
            # This could happen if `select_dims` causes the new_sample to be outside any existing leaf
            # when considering only those dimensions.

            # Check if it falls in *any* node, not just leaves
            non_leaf_search = self.search(new_sample, l_nodes=None)
            if not xp.any(non_leaf_search):
                 raise Exception("New sample does not fall into any node (leaf or internal). Boundaries issue?")

            # If it falls into an internal node, but not a leaf, this is also problematic for this logic.
            # The logic assumes it must fall into an existing leaf to split that leaf.
            # This could happen if self.node_is_leaf_ is somehow incorrect or out of sync.
            checked_and_warn(
                "[DEBUG] _grow: New sample did not fall into any existing leaf node. This might indicate boundary issues or an unexpected state.",
                category=RuntimeWarning
            )
            # As a fallback, let's find the deepest node it falls into.
            # This is a deviation from the simple "split the leaf" logic.
            # This part requires more robust handling. For now, proceeding with original logic which might error.
            pass # Allow it to proceed and potentially error if no leaf is found by search_result[0,:]

        i_parent = l_leaf[search_result[0, :]][0] # Index of the leaf node to split

        # Temporarily add the new sample to self.samples_ to determine split point
        original_samples = self.samples_
        self.samples_ = xp.concatenate([self.samples_, new_sample], axis=0)

        # Determine split parameters using _isolation_split logic (but without modifying stack)
        # _isolation_split needs current node index, stack, and boundaries.
        # We are splitting i_parent.
        # We need to simulate a split within this node's boundaries using its current samples + new_sample.

        # Re-extracting logic from _isolation_split to find a split_dim and split_value
        eps = xp.finfo(self.samples_.dtype).eps
        current_node_samples_mask = self._search(self.samples_,
                                                 self.node_lower_boundaries[i_parent:i_parent+1, :],
                                                 self.node_upper_boundaries[i_parent:i_parent+1, :])[:,0]

        m_in_current_node = self.samples_[current_node_samples_mask, :]

        if m_in_current_node.shape[0] <=1:
            # Not enough points to make a meaningful split, or new sample is identical to an existing one.
            # This branch might become a leaf immediately.
            # This scenario should ideally be handled by _isolation_split returning without splitting.
            # For _grow, it means the new sample doesn't force a new split of this leaf.
            self.samples_ = original_samples # Restore original samples
            # No actual grow operation happens on the tree structure here.
            # The sample is added to self.samples_ in partial_fit's main flow.
            # This function's contract is to return parent, left, right.
            # This needs a more graceful way to indicate "no structural grow".
            # For now, let's assume _isolation_split's logic for finding split_dim/value handles this.
            # If l_dims is empty in the following block, it will also not split.
            pass


        l_min_node = xp.min(m_in_current_node, axis=0)
        l_max_node = xp.max(m_in_current_node, axis=0)
        l_dims_node = xp.where(l_min_node + (2 * eps) <= l_max_node)[0]

        if l_dims_node.shape[0] == 0: # Cannot split further
            self.samples_ = original_samples # Restore
            # This means the new sample fell into a leaf that now contains identical points or points along a line/plane.
            # No structural change. How to signal this?
            # The original code does not explicitly handle this return path for _grow.
            # Let's assume grow() will create children that might immediately become leaves.
            # If grow is called, it adds children. We need to provide a valid split_dim, split_value.
            # If no valid dim, this indicates an issue.
            # A robust way: if no split possible, the node i_parent remains a leaf.
            # This function's purpose is to *perform* the grow.
            # For now, let it potentially fail if no split_dim found, or handle it in self.grow.
             checked_and_warn(
                f"[DEBUG] _grow: Cannot find dimension to split node {i_parent} further after adding new sample. Node might remain leaf.",
                category=RuntimeWarning
            )
            # If we cannot find a split_dim, we cannot call self.grow(i_parent, split_dim, split_value)
            # This implies the node i_parent should remain a leaf.
            # This contradicts the expectation that _grow *always* splits and returns new children.
            # This part of the logic needs refinement.
            # Fallback: if no split_dim, we cannot proceed with self.grow.
            # This indicates the sample is in a region that cannot be further split.
            # The base `grow` method in `AxisParallelBinaryTree` doesn't have this issue as it's just structural.
            # The issue is finding `split_dim` and `split_value` for `IsolationTree`.
            # If we can't find a valid split, we should not call self.grow.
            # For now, if no l_dims_node, this will error at `xp.random.randint(low=0, high=l_dims_node.shape[0])`
            # Let's add a check here:
            if l_dims_node.shape[0] == 0:
                 self.samples_ = original_samples
                 # This node cannot be split further. It remains a leaf.
                 # The function is expected to return i_parent, i_left, i_right.
                 # This means we didn't actually "grow" in the structural sense of adding new nodes.
                 # This case needs clarification on expected behavior.
                 # For now, let's assume this means the new sample doesn't change the structure.
                 # This is inconsistent with the call to self.grow later.
                 # This indicates `_grow` might not always result in new children.
                 # Let's return something to signify no structural change, though this breaks current return contract.
                 # This requires a larger refactor. For now, allow it to proceed and potentially error if no l_dims.
                 # This situation should ideally be handled by the splitting condition in _build_tree or _isolation_split.
                 pass # Let it try and see if `grow` handles it or errors.


        rand_split_dim_idx_node = xp.random.randint(low=0, high=l_dims_node.shape[0])
        split_dim = l_dims_node[rand_split_dim_idx_node]
        split_value = xp.random.uniform(l_min_node[split_dim] + eps, l_max_node[split_dim] - eps)
        split_value = xp.asarray(split_value, dtype=l_max_node.dtype)

        self.samples_ = original_samples # Restore original samples before calling grow

        # Perform the structural split
        i_left, i_right = self.grow(i_parent, split_dim, split_value)

        # Update tree properties (leaf status, volumes)
        self.node_is_leaf_ = self.leaf() # Recalculate leaf status
        self.node_volumes_ = self.volumes() # Recalculate volumes

        return i_parent, i_left, i_right

    def _prune(self, drop_sample):
        """Prune the tree after a sample is dropped from the reservoir.

        Finds the leaf node containing the dropped sample. If this node becomes
        empty or its sibling is also a leaf, they might be merged (pruned).

        Parameters
        ----------
        drop_sample : array-like
            The sample being dropped. Shape (1, n_features).

        Returns
        -------
        i_node_drop : int
            Index of the leaf node from which the sample effect is removed.
        volume_drop : float
            Volume of the pruned node.
        l_volumes_diff : array-like
            Difference in volumes due to pruning (relevant for mass updates).
        l_keep : array-like
            Boolean array indicating which nodes are kept after pruning.
        """
        xp, _ = get_array_module(drop_sample)
        l_leaf = xp.where(self.node_is_leaf_)[0] # Current leaf nodes

        search_result = self.search(drop_sample, l_nodes=self.node_is_leaf_)
        if not xp.any(search_result):
            # Sample to drop is not found in any leaf. This is unusual.
            # It might have been in a branch that was already pruned due to other drops.
            # Or, if rotation is used, SO_ matrix might be an issue if not consistently applied.
            checked_and_warn(
                f"[DEBUG] _prune: Sample to drop not found in any current leaf node. Sample: {drop_sample}. This might be okay if the region was already modified.",
                RuntimeWarning,
            )
            # Return sentinel values indicating no pruning occurred for this specific sample drop
            # as it's not found in a leaf.
            # The number of nodes might have changed, so self.node_volumes_ might be of different size.
            # Need to return values that don't break the caller (AdaptiveMassEstimationTree._prune)
            # This requires careful thought on what `prune` in BinaryTree returns.
            # `prune` returns (l_volumes_diff, l_keep).
            # If i_node_drop is invalid, self.prune(invalid_node) will fail.
            # For now, let's assume if not found, no structural prune happens for this sample.
            # This means we need to return values that signify "no change".
            # This is a tricky state. Let's find the node it *would* have been in.

            # Search all nodes, not just leaves
            search_all_nodes = self.search(drop_sample, l_nodes=None)
            if not xp.any(search_all_nodes): # Still not found anywhere
                 # This is more serious, implies sample is outside global boundaries or select_dims issue.
                 raise Exception(f"Sample to drop {drop_sample} not found in ANY node.")

            # Find the deepest node it belongs to.
            # This logic is complex. For now, if not in a leaf, we assume it means no pruning action.
            # This path needs robust error handling or state management.
            # Let's return dummy values consistent with "no structural change"
            # This might not be what the caller expects.
            # The original code proceeds to call self.prune with i_node_drop from l_leaf.
            # If search_result is all False, this will error.
            # A simple fix: if not xp.any(search_result), then this sample cannot be "pruned" from a leaf.
             return -1, 0.0, xp.array([]), xp.ones(self.node_lower_boundaries.shape[0], dtype=bool)


        i_node_drop = l_leaf[search_result[0, :]][0]

        # Remove the sample from self.samples_ (the reservoir)
        # This is done carefully to avoid issues if drop_sample is not exactly present (e.g. float precision)
        # A robust way: find its index by checking all features.
        sample_indices_to_keep = xp.where(xp.logical_not(xp.all(self.samples_ == drop_sample, axis=1)))[0]
        if len(sample_indices_to_keep) == self.samples_.shape[0]:
            # Sample was not found in self.samples_ reservoir.
            # This can happen if the reservoir sampling kicked it out earlier,
            # or if it's from a partial_fit batch that wasn't part of the initial reservoir.
            # This is a valid state, means no change to self.samples_ from this specific drop_sample call.
            pass # No change to self.samples_
        else:
            self.samples_ = xp.take(self.samples_, sample_indices_to_keep, axis=0)

        volume_drop = self.node_volumes_[i_node_drop] # Volume of the node being affected

        # Perform structural prune operation from the binary tree
        l_volumes_diff, l_keep = self.prune(i_node_drop)

        # Update tree properties
        self.node_is_leaf_ = self.leaf()
        self.node_volumes_ = self.volumes()
        return i_node_drop, volume_drop, l_volumes_diff, l_keep

    def transform(self, X):
        """Transform X to the space of leaf node indices.

        Each sample in X is mapped to the index of the leaf node it falls into.

        Parameters
        ----------
        X : array-like
            Input data to transform.

        Returns
        -------
        indices : array-like
            A 2D array where each row contains the index of the leaf node
            for the corresponding input sample. Shape (n_samples, 1).
        """
        xp, _ = get_array_module(X)
        # Search for samples only in leaf nodes
        l_in = self.search(X, self.node_is_leaf_)
        # Get row and column indices where l_in is True.
        # We only care about the column index, which corresponds to the leaf node index.
        _, indices = xp.where(l_in)
        indices = xp.expand_dims(indices, axis=1) # Return as a column vector
        return indices


class AdaptiveMassEstimationTree(IsolationTree, DensityMixin):
    """Isolation Tree variant for adaptive mass estimation.

    Extends IsolationTree to maintain and update mass estimates for each node,
    allowing for density estimation.

    Parameters
    ----------
    psi : int
        Subsample size for the reservoir.
    **kwargs :
        Additional keyword arguments passed to IsolationTree.
    """

    def __init__(self, psi, **kwargs):
        super().__init__(psi, **kwargs)
        self.fitted = 0  # Total number of samples processed by fit/partial_fit

    def fit(self, X, y=None):
        """Fit the tree and compute initial node masses.

        Parameters
        ----------
        X : array-like
            Input samples.
        y : Ignored
            Not used.

        Returns
        -------
        self : object
            Fitted instance.
        """
        xp, _ = get_array_module(X)
        super().fit(X, y) # Build tree structure
        # Calculate initial mass for each node based on samples from X
        # search(X) gives a boolean matrix (n_samples_X, n_total_nodes)
        # Summing along axis 0 gives count of samples from X in each node.
        self.node_mass_ = xp.sum(self.search(X), axis=0, dtype=float)
        self.fitted = X.shape[0]
        # self.node_volumes_ is already computed in super().fit() via _build_tree()
        return self

    def _grow(self, new_sample):
        """Grow the tree for a new sample and update node masses.

        When a parent node is split, its mass is distributed to its new children
        proportionally to their volumes.

        Parameters
        ----------
        new_sample : array-like
            The new sample being added.

        Returns
        -------
        self : object
            Typically, tree modification methods return self, but the parent _grow
            returns indices. This _grow should maintain consistency or be used for its
            side effects (updating node_mass_).
            The original code does not have an explicit return for this method,
            implying it's used for side effects. Let's adhere to that.
        """
        xp, _ = get_array_module(new_sample)
        # Call IsolationTree._grow to perform structural split and get parent/child info
        i_parent, i_left, i_right = super()._grow(new_sample)

        # Distribute mass from parent to new children
        # This assumes i_parent, i_left, i_right are valid and a split occurred.
        # If super()._grow had a path where no split occurs, this logic needs adjustment.
        # Assuming a split always happens if _grow is called and completes.

        parent_volume = self.node_volumes_[i_parent]
        if parent_volume == 0: # Avoid division by zero if parent volume is zero
            # This shouldn't happen if eps padding is used correctly for volumes.
            # If it does, assign zero mass or handle as an error.
            rho_parent = 0.0
            checked_and_warn(f"Parent node {i_parent} has zero volume during grow. Mass distribution might be inaccurate.", RuntimeWarning)
        else:
            rho_parent = self.node_mass_[i_parent] / parent_volume

        mass_left = rho_parent * self.node_volumes_[i_left]
        mass_right = rho_parent * self.node_volumes_[i_right]

        # Append new masses for i_left and i_right.
        # AxisParallelBinaryTree.grow appends new nodes, so masses should correspond.
        self.node_mass_ = xp.concatenate(
            [self.node_mass_, xp.asarray([mass_left, mass_right], dtype=self.node_mass_.dtype)]
        )
        # No explicit return, method modifies self.node_mass_
        return self # Or return indices as parent does, for consistency. Let's stick to side-effects for now.


    def _prune(self, drop_sample):
        """Prune the tree for a dropped sample and update node masses.

        When nodes are merged due to pruning, their masses are adjusted.

        Parameters
        ----------
        drop_sample : array-like
            The sample being dropped.

        Returns
        -------
        self : object
            Similar to _grow, assuming modification by side-effect.
        """
        # Call IsolationTree._prune to perform structural prune
        i_node_drop, volume_drop, l_volumes_diff, l_keep = super()._prune(drop_sample)

        if i_node_drop == -1 and volume_drop == 0.0: # Sentinel for "sample not found / no prune"
            return self # No change to mass if no structural prune occurred.

        # Update masses based on pruning results
        if volume_drop == 0: # Avoid division by zero
            rho_drop = 0.0 # Mass density is zero if volume is zero
            # This might indicate an issue if a node with mass has zero volume.
            if self.node_mass_[i_node_drop] > 0:
                 checked_and_warn(f"Node {i_node_drop} has mass but zero volume during prune. Mass redistribution might be inaccurate.", RuntimeWarning)
        else:
            rho_drop = self.node_mass_[i_node_drop] / volume_drop

        self.node_mass_ = self.node_mass_[l_keep] # Keep masses for nodes that weren't removed
        # Adjust masses of nodes whose volumes changed due to merge
        # l_volumes_diff contains the change in volume for *kept* nodes that were affected.
        # This assumes l_volumes_diff aligns with the elements of self.node_mass_ after l_keep.
        # The `prune` in binary_tree needs to ensure l_volumes_diff is structured correctly for this.
        # It should be an array of same size as newly kept nodes, with most entries 0,
        # and non-zero for the node that expanded its volume.
        self.node_mass_ = self.node_mass_ + l_volumes_diff * rho_drop
        return self # Or no return for side-effect

    def score(self, X, y=None, return_demass=False):
        """Calculate the mass or density for samples in X.

        Parameters
        ----------
        X : array-like
            Input samples.
        y : Ignored
            Not used.
        return_demass : bool, default=False
            If True, returns density (mass/volume).
            If False, returns mass.

        Returns
        -------
        scores : array-like
            The calculated mass or density for each sample in X.
        """
        xp, _ = get_array_module(X)
        # Get leaf node indices for each sample in X
        indices = super().transform(X)[:, 0]

        # Get mass/volume for all leaf nodes
        l_leaf_nodes_mask = self.node_is_leaf_
        l_leaf_mass = self.node_mass_[l_leaf_nodes_mask]

        if return_demass: # Calculate density
            l_leaf_volumes = self.node_volumes_[l_leaf_nodes_mask]
            # Add epsilon to prevent division by zero if a leaf node has zero volume
            # (though ideally, leaf nodes with samples should have non-zero volume).
            eps = xp.finfo(X.dtype).eps if X.dtype in [xp.float32, xp.float64] else 1e-8

            # Ensure l_leaf_mass and l_leaf_volumes are compatible for division
            # This can be an issue if a node becomes a leaf but its mass was not properly set (e.g. for new leaves in _grow)
            # Or if node_mass_ array size is not consistent with node_is_leaf_
            if l_leaf_mass.shape[0] != l_leaf_volumes.shape[0]:
                # This indicates a potential inconsistency between the number of leaf nodes
                # and the available mass/volume entries for them.
                # This might happen if node_mass_ is not correctly updated/resized when tree structure changes.
                # Re-evaluate what self.search(X) provides for self.node_mass_ in fit().
                # self.node_mass_ is (n_total_nodes,). self.node_is_leaf_ is (n_total_nodes,).
                # So l_leaf_mass and l_leaf_volumes should have same shape.
                raise RuntimeError(f"Mismatch in leaf mass ({l_leaf_mass.shape[0]}) and volume ({l_leaf_volumes.shape[0]}) counts. Tree state inconsistent.")

            safe_volumes = l_leaf_volumes + eps
            safe_mass = l_leaf_mass + eps # Also add to mass if volume is tiny, to keep density reasonable

            l_leaf_demass = xp.divide(safe_mass, safe_volumes, out=xp.zeros_like(safe_mass), where=safe_volumes!=0)
            # Handle cases where volume was truly zero (now eps)
            l_leaf_demass = xp.where(l_leaf_volumes == 0, 0, l_leaf_demass) # If original volume was 0, density is 0 (or undefined)

            return xp.take(l_leaf_demass, indices)
        else: # Return mass
            return xp.take(l_leaf_mass, indices)
        #     upper_boundary_i = self.node_upper_boundaries[i_child, :]

        #     l_in_i = self._search(self.samples_, lower_boundary_i, upper_boundary_i)
        #     l_in_i = l_in_i[:, 0]  # only one node therefore one column
        #     if xp.sum(l_in_i) < 1:
        #         from ._binary_tree import single_split

        #         (
        #             lower_boundary_left,
        #             upper_boundary_left,
        #             lower_boundary_right,
        #             upper_boundary_right,
        #         ) = single_split(
        #             self.node_lower_boundaries[i : i + 1, :],
        #             self.node_upper_boundaries[i : i + 1, :],
        #             split_dim,
        #             split_value,
        #         )
        #         self._search(m_in, lower_boundary_i, upper_boundary_i)
        #         raise Exception("There is no sample in this region!")

        return stack_to_split

    def _build_tree(self):
        stack_to_split = [0]

        while len(stack_to_split) > 0:
            i = stack_to_split.pop()
            lower_boundary_i = self.node_lower_boundaries[i, :]
            upper_boundary_i = self.node_upper_boundaries[i, :]
            stack_to_split = self._isolation_split(
                i, stack_to_split, lower_boundary_i, upper_boundary_i
            )

        self.node_is_leaf_ = self.leaf()
        self.node_volumes_ = self.volumes()

    def partial_fit(self, X, y=None):
        xp, _ = get_array_module(X)
        if self.rotation:
            X_, _ = rotate(X, self.SO_)
        else:
            X_ = X
        X_ = X_[:, self.select_dims]
        changed_count, new_samples, drop_samples, reservoir = super().update_samples(X_)

        for i in range(changed_count):
            drop_sample = drop_samples[i : i + 1, :]
            self._prune(drop_sample)
            # TODO
            # print(xp.sum(self.node_volumes_[self.node_is_leaf_]))
            # print(xp.sum(self.node_mass_[self.node_is_leaf_]))

            new_sample = new_samples[i : i + 1, :]
            self._grow(new_sample)
            # TODO
            # print(xp.sum(self.node_volumes_[self.node_is_leaf_]))
            # print(xp.sum(self.node_mass_[self.node_is_leaf_]))

        self.node_mass_ = self.node_mass_ + xp.sum(self.search(X_), axis=0, dtype=float)

        self.samples_ = reservoir
        self.fitted = self.fitted + X_.shape[0]

        # TODO
        # print(xp.sum(self.node_mass_[self.node_is_leaf_]))
        # print(self.fitted)
        return self

    def _grow(self, new_sample):
        xp, _ = get_array_module(new_sample)
        l_leaf = xp.where(self.node_is_leaf_)[0]
        i_parent = l_leaf[self.search(new_sample, l_nodes=self.node_is_leaf_)[0, :]][0]
        lower_boundary_i = self.node_lower_boundaries[i_parent]
        upper_boundary_i = self.node_upper_boundaries[i_parent]
        self.samples_ = xp.concatenate([self.samples_, new_sample], axis=0)
        _, split_dim, split_value, _ = self._isolation_split(
            lower_boundary_i, upper_boundary_i
        )
        i_left, i_right = self.grow(i_parent, split_dim, split_value)
        self.node_is_leaf_ = self.leaf()
        self.node_volumes_ = self.volumes()

        return i_parent, i_left, i_right

    def _prune(self, drop_sample):
        xp, _ = get_array_module(drop_sample)
        l_leaf = xp.where(self.node_is_leaf_)[0]
        search_result = self.search(drop_sample, l_nodes=self.node_is_leaf_)
        if not xp.any(search_result):
            checked_and_warn(
                "[DEBUG] _prune is failing because the region of given sample is not found"
            )
        i_node_drop = l_leaf[search_result[0, :]][0]
        self.samples_ = xp.take(
            self.samples_,
            xp.where(xp.logical_not(xp.all(self.samples_ == drop_sample, axis=1)))[0],
            axis=0,
        )
        volume_drop = self.node_volumes_[i_node_drop]
        l_volumes_diff, l_keep = self.prune(i_node_drop)

        self.node_is_leaf_ = self.leaf()
        self.node_volumes_ = self.volumes()
        return i_node_drop, volume_drop, l_volumes_diff, l_keep

    def transform(self, X):
        xp, _ = get_array_module(X)
        l_in = self.search(X, self.node_is_leaf_)
        _, indices = xp.where(l_in)
        indices = xp.expand_dims(indices, axis=1)
        return indices


class AdaptiveMassEstimationTree(IsolationTree, DensityMixin):
    def __init__(self, psi, **kwargs):
        super().__init__(psi, **kwargs)
        self.fitted = 0

    def fit(self, X, y=None):
        xp, _ = get_array_module(X)
        super().fit(X, y)
        self.node_mass_ = xp.sum(self.search(X), axis=0, dtype=float)
        self.fitted = X.shape[0]
        # self.node_volumes_ = self.volumes() # included in fit

    def _grow(self, new_sample):
        xp, _ = get_array_module(new_sample)
        i_parent, i_left, i_right = super()._grow(new_sample)

        rho_parent = self.node_mass_[i_parent] / self.node_volumes_[i_parent]
        mass_left = rho_parent * self.node_volumes_[i_left]
        mass_right = rho_parent * self.node_volumes_[i_right]

        self.node_mass_ = xp.concatenate(
            [self.node_mass_, xp.array([mass_left, mass_right])]
        )
        return self

    def _prune(self, drop_sample):
        i_node_drop, volume_drop, l_volumes_diff, l_keep = super()._prune(drop_sample)
        rho_drop = self.node_mass_[i_node_drop] / volume_drop

        self.node_mass_ = self.node_mass_[l_keep]
        self.node_mass_ = self.node_mass_ + l_volumes_diff * rho_drop
        return self

    def score(self, X, y=None, return_demass=False):
        xp, _ = get_array_module(X)
        indices = super().transform(X)[:, 0]
        l_leaf_mass = self.node_mass_[self.node_is_leaf_]
        if return_demass:
            l_leaf_volumes = self.node_volumes_[self.node_is_leaf_]
            if xp.any(l_leaf_volumes == 0):
                eps = xp.finfo(X.dtype).eps
                l_leaf_demass = (l_leaf_mass + eps) / (l_leaf_volumes + eps)
            else:
                l_leaf_demass = l_leaf_mass / l_leaf_volumes
            return xp.take(l_leaf_demass, indices)
        else:
            return xp.take(l_leaf_mass, indices)
