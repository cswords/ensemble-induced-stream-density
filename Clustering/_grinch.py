import time

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import ClusterMixin, BaseEstimator

# from sklearn.utils.validation import check_is_fitted


class Grinch(ClusterMixin, BaseEstimator):
    def __init__(
        self,
        n_clusters=-1,
        rotate_cap=100,
        graft_cap=100,
        norm="l2",
        sim="dot",
        max_num_points=None,  # 1000000,
        pids=None,
        canopies=None,
    ):
        self.n_clusters = n_clusters

        self.point_counter = 0

        # the representation secures max_num_points positions for the points
        # max_nodes is the overall number of nodes.

        if max_num_points is not None:
            self.max_num_points = max_num_points
            self.max_nodes = self.max_num_points * 3
        else:
            self.max_num_points = None  # will be initalised in fit

        self.norm = norm

        self.pids = pids
        self.canopies = canopies

        self.sibs = None
        self.rotate_cap = rotate_cap
        self.graft_cap = graft_cap
        self.perform_graft = True

        self.init_compute_centroid(norm)
        self.init_csim(sim)

        # Timing and stats
        self.time_in_search = 0
        self.time_in_rotate = 0
        self.time_in_update = 0
        self.time_in_update_walk = 0
        self.time_in_update_from_children = 0
        self.time_in_graft_score_only = 0
        self.time_in_graft = 0
        self.time_in_lca = 0
        self.time_in_descendants = 0
        self.time_in_centroid = 0
        self.time_in_graft_search = 0
        self.time_in_graft_get_scores = 0
        self.this_time_in_graft_get_scores = 0

        self.number_of_rotates = 0
        self.number_of_grafts = 0
        self.number_of_grafts_considered = 0
        self.number_of_graft_allowable = 0
        self.number_of_rotates_considered = 0

        self.this_number_of_rotates = 0
        self.this_number_of_grafts = 0
        self.this_number_of_grafts_considered = 0
        self.this_number_of_rotates_considered = 0
        self.this_time_in_search = 0
        self.this_time_in_rotate = 0
        self.this_time_in_update = 0
        self.this_time_in_graft = 0
        self.this_time_in_graft_search = 0
        self.cached_nns = None

        self.fitted = 0

    def init_grinch(self):
        self.centroids = np.zeros((self.max_nodes, self.dim_), dtype=np.float32)
        self.sums = np.zeros((self.max_nodes, self.dim_), dtype=np.float32)

        self.children = [[] for _ in range(self.max_nodes)]
        self.descendants = [[] for _ in range(self.max_nodes)]
        self.scores = -np.inf * np.ones(self.max_nodes, dtype=np.float32)
        self.needs_update_model = np.zeros(self.max_nodes, dtype=np.bool_)
        self.new_node = np.ones(self.max_nodes, dtype=np.bool_)
        self.needs_update_desc = np.zeros(self.max_nodes, dtype=np.bool_)
        self.parent = -1 * np.ones(self.max_nodes, dtype=np.int32)
        self.next_node_id = self.max_num_points
        self.num_descendants = -1 * np.ones(self.max_nodes, dtype=np.float32)

    def fit(self, X, y=None, shuffle=True):
        if shuffle:
            shuffle_idx = np.arange(X.shape[0])
            np.random.shuffle(shuffle_idx)
            self.points_ = X[shuffle_idx, :]
        else:
            self.points_ = X
        self.num_points_ = X.shape[0]
        self.dim_ = X.shape[1]

        if self.max_num_points is None:
            self.max_num_points = self.num_points_ * 3
            self.max_nodes = self.max_num_points * 3

        self.init_grinch()
        self.build_dendrogram()
        self.fitted += X.shape[0]
        return self

    def partial_fit(self, X, y=None):
        self.points_ = np.vstack([self.points_, X,])
        i_start = self.num_points_
        self.num_points_ += X.shape[0]

        for i in range(i_start, self.num_points_):
            self.stats_string()
            self.insert(i)
        self.fitted += X.shape[0]
        return self

    def predict(self, X=None, y=None, k=None):
        if k == None:
            k = self.n_clusters
        self.labels_ = self.k_clustering(k)
        return self.labels_
        # if X is None:
        #     return self.labels_
        # else:
        #     X_idx = np.where(
        #         np.all(
        #             np.expand_dims(self.points_, axis=1) == np.expand_dims(X, axis=0),
        #             axis=2,
        #         )
        #     )[0]
        #     if X_idx.shape[0] != X.shape[0]:
        #         print("shit")
        #     return self.labels_[X_idx]

    def fit_predict(self, X, y=None, k=None, shuffle=True):
        return self.fit(X, shuffle=shuffle).predict(k=k)

    def all_valid_nodes(self):
        r = self.root()
        return [x for x in range(self.next_node_id) if x == r or self.parent[x] >= 0]

    def all_valid_internal_nodes(self):
        r = self.root()
        return [
            x
            for x in range(self.num_points_, self.next_node_id)
            if (x == r or self.parent[x] >= 0)
        ]

    def clear_stats(self):
        self.this_number_of_rotates = 0
        self.this_number_of_grafts = 0
        self.this_number_of_grafts_considered = 0
        self.this_number_of_rotates_considered = 0
        self.this_time_in_search = 0
        self.this_time_in_rotate = 0
        self.this_time_in_update = 0
        self.this_time_in_graft = 0
        self.this_time_in_graft_search = 0
        self.this_time_in_graft_get_scores = 0

    def stats_string(self):
        r = (
            "search_time=%s\trotate_time=%s\tgraft_time=%s\tupdate_time=%s\tdescendant_time=%s\tcentroid_time=%s\tlca_time=%s\tnum_rotate=%s\tnum_graft=%s\tnum_rotate_considered=%s\tnum_graft_considered=%s\t%s\n"
            % (
                self.time_in_search,
                self.time_in_rotate,
                self.time_in_graft,
                self.time_in_update,
                self.time_in_descendants,
                self.time_in_centroid,
                self.time_in_lca,
                self.number_of_rotates,
                self.number_of_grafts,
                self.number_of_rotates_considered,
                self.number_of_grafts_considered,
                self.this_stats_string(),
            )
        )
        self.clear_stats()
        return r

    def this_stats_string(self):
        r = (
            "last_stats\tnum_rotate=%s\tnum_graft=%s\tnum_rotate_considered=%s\tnum_graft_considered=%s\n"
            % (
                self.this_number_of_rotates,
                self.this_number_of_grafts,
                self.this_number_of_rotates_considered,
                self.this_number_of_grafts_considered,
            )
        )
        return r

    def lca_and_ancestors(self, i, j):
        s = time.time()
        if i == j:
            return (i, [], [])
        if self.parent[i] == -1:
            # print("lca_and_ancestors i = root %s", i)
            return (i, [], [])
        curr_node = j
        thisAnclist = self.get_ancs_with_self(i)
        thisAnc = dict([(nid, idx) for idx, nid in enumerate(thisAnclist)])
        other2lca = []
        while curr_node not in thisAnc:
            other2lca.append(curr_node)
            curr_node = self.get_parent(curr_node)
        this2lca = thisAnclist[: thisAnc[curr_node]]
        self.time_in_lca += time.time() - s
        return (
            curr_node,
            [x for x in this2lca if self.num_descendants[x] < self.graft_cap],
            [x for x in other2lca if self.num_descendants[x] < self.graft_cap],
        )

    def find_rotate(self, gnode, sib):
        s = time.time()
        # print("[rotate] find_rotate(%s, %s)", gnode, sib)
        curr = sib
        score = self.e_score(gnode, sib)
        curr_parent = self.get_parent(curr)
        curr_parent_score = (
            -np.Inf if curr_parent == -1 else self.get_score(curr_parent)
        )
        while (
            curr_parent != -1
            and score < curr_parent_score
            and self.num_descendants[curr_parent] < self.rotate_cap
        ):
            # print(
            #     "[rotate] curr %s curr_parent %s gnode %s score %s curr_parent_score %s",
            #     curr,
            #     curr_parent,
            #     gnode,
            #     score,
            #     curr_parent_score,
            # )
            curr = curr_parent
            curr_parent = self.get_parent(curr)
            curr_parent_score = (
                -np.Inf if curr_parent == -1 else self.get_score(curr_parent)
            )
            score = self.e_score(gnode, sib)
            self.number_of_rotates += 1
            self.this_number_of_rotates += 1
            self.number_of_rotates_considered += 1
            self.this_number_of_rotates_considered += 1
        # print("[rotate] find_rotate(%s, %s) = %s", gnode, sib, curr)
        self.time_in_rotate += time.time() - s
        self.this_time_in_rotate += time.time() - s
        return curr

    def get_centroid_batch(self, i):
        return self.centroids[i]

    def get_centroid(self, i):
        return np.expand_dims(self.centroids[i], 0)

    def init_compute_centroid(self, norm):
        if norm == "l2":
            # print("Using centroid = l2")
            self.compute_centroid = self.compute_centroid_l2_norm
        elif norm == "l_inf":
            # print("Using centroid = l_inf")
            self.compute_centroid = self.compute_centroid_l_inf_norm
        elif norm == "none":
            # print("Using centroid = none")
            self.compute_centroid = self.compute_centroid_no_norm

    def compute_centroid_l2_norm(self, i):
        # if the node is new we don't need to zero
        if not self.new_node[i]:
            self.centroids[i] *= 0
        self.centroids[i] += self.sums[i]
        self.centroids[i] /= self.num_descendants[i]
        if type(i) is np.array:
            norms = np.linalg.norm(self.centroids[i], axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
        else:
            norms = np.linalg.norm(self.centroids[i])
            norms = norms if norms > 0 else 1.0
        self.centroids[i] /= norms

    def compute_centroid_no_norm(self, i):
        # if the node is new we don't need to zero
        if not self.new_node[i]:
            self.centroids[i] *= 0
        self.centroids[i] += self.sums[i]
        self.centroids[i] /= self.num_descendants[i]

    def compute_centroid_l_inf_norm(self, i):
        # if the node is new we don't need to zero
        if not self.new_node[i]:
            self.centroids[i] *= 0
        self.centroids[i] += self.sums[i]
        self.centroids[i] /= self.num_descendants[i]
        self.centroids[i] /= np.linalg.norm(self.centroids[i], np.inf)

    def get_sum(self, i):
        return np.expand_dims(self.sums[i], 0)

    def set_sum(self, i, v):
        self.sums[i] = v

    def update_desc(self, i, use_tqdm=False):
        s = time.time()
        needs_update = []
        to_check = [i]
        while to_check:
            curr = to_check.pop(0)
            if self.needs_update_desc[curr]:
                needs_update.append(curr)
                for c in self.get_children(curr):
                    to_check.append(c)
        self.time_in_update_walk += time.time() - s
        for j in range(len(needs_update) - 1, -1, -1):
            self.single_update_desc(needs_update[j])
        self.time_in_update += time.time() - s

    def single_update_desc(self, i):
        s = time.time()
        # print("updating node %s", i)
        assert self.needs_update_desc[i]
        kids = self.get_children(i)
        self.descendants[i].clear()
        for kid in kids:
            self.descendants[i] += np.unique(self.descendants[kid]).tolist()
        self.time_in_descendants += time.time() - s
        self.new_node[i] = False
        self.needs_update_desc[i] = False

    def get_descendants(self, i):
        if self.needs_update_desc[i]:
            # print("Updating because of get_descendants!")
            self.update_desc(i)
        return self.descendants[i]

    def graft(self, gnode):
        s = time.time()
        # print("[graft] graft(%s)", gnode)
        curr = gnode

        # Find offlimits
        offlimits1 = self.get_descendants(curr)
        if (
            self.get_parent(curr) != -1
            and len(self.get_children(self.get_sibling(curr))) == 0
        ):
            offlimits2 = [self.get_sibling(curr)]
        else:
            offlimits2 = []
        # print(
        #     "[graft] len(offlimits1)=%s len(offlimits2)=%s",
        #     len(offlimits1),
        #     len(offlimits2),
        # )
        # print("[graft] offlimits1 %s offlimits2 %s", str(offlimits1), str(offlimits2))

        # Find updates
        self.update(curr)
        curr_v = self.get_centroid(curr)

        # Do search
        search_st = time.time()
        _, nns = self.cknn(curr_v, offlimits1, offlimits2)
        self.time_in_graft_search += time.time() - search_st
        self.this_time_in_graft_search += time.time() - search_st
        if len(nns) == 0:
            # print("No nearest neighbors after nns....")
            return

        oneNN = nns[0]
        # print("Nearest neighbor is %s", oneNN)
        # lca, this2anc, other2anc = self.lca_and_ancestors(gnode, oneNN)
        _, this2anc, other2anc = self.lca_and_ancestors(gnode, oneNN)
        # print(
        #     "lca %s len(this2anc) %s len(other2anc) %s",
        #     lca,
        #     len(this2anc),
        #     len(other2anc),
        # )
        if this2anc and other2anc:
            # find all pairwise distances
            # this_vecs = self.get_centroid(this2anc)
            # anc_vecs = self.get_centroid(other2anc)

            # M by N
            M = len(this2anc)
            N = len(other2anc)
            score_if_grafted = self.e_score_batch(this2anc, other2anc)
            assert score_if_grafted.shape[0] == M
            assert score_if_grafted.shape[1] == N
            # 1 by N
            nn_parent_score = np.expand_dims(
                self.get_score_batch(self.get_parent(other2anc)), 0
            )
            assert nn_parent_score.shape[0] == 1
            assert nn_parent_score.shape[1] == N
            # M by 1
            curr_parent_score = np.expand_dims(
                self.get_score_batch(self.get_parent(this2anc)), 1
            )
            assert curr_parent_score.shape[0] == M
            assert curr_parent_score.shape[1] == 1

            not_i_like_you = score_if_grafted <= curr_parent_score
            not_you_like_me = score_if_grafted <= nn_parent_score
            assert not_i_like_you.shape[0] == M
            assert not_i_like_you.shape[1] == N
            assert not_you_like_me.shape[0] == M
            assert not_you_like_me.shape[1] == N

            graft_condition = not_i_like_you | not_you_like_me
            num_meeting_condition = graft_condition.sum()
            self.number_of_graft_allowable += num_meeting_condition
            # total_candidate_grafts = max(1.0, len(this2anc) * len(other2anc))

            score_if_grafted[graft_condition] = 0
            argmax = np.argmax(score_if_grafted)
            argmax_row = int(argmax / score_if_grafted.shape[1])
            argmax_col = argmax % score_if_grafted.shape[1]
            best_1 = this2anc[argmax_row]
            best_2 = other2anc[argmax_col]
            if (
                not not_i_like_you[argmax_row, argmax_col]
                and not not_you_like_me[argmax_row, argmax_col]
            ):
                self.number_of_grafts += 1
                self.this_number_of_grafts += 1
                bestPair2gp = self.get_parent(self.get_parent(best_2))
                parent = self.node_from_nodes(best_1, best_2)
                self.make_sibling(best_1, best_2, parent)
                # print(
                #     "[graft] node %s grafts node %s, scores %s > max(%s, %s)"
                #     % (
                #         best_1,
                #         best_2,
                #         score_if_grafted[argmax_row, argmax_col],
                #         curr_parent_score[argmax_row, 0],
                #         nn_parent_score[0, argmax_col],
                #     )
                # )
                for start in [bestPair2gp, self.get_parent(curr)]:
                    curr_update = start
                    while curr_update != -1:
                        self.updated_from_children(curr_update)
                        curr_update = self.get_parent(curr_update)
            # else:
            #     print("[graft] There was no case where we wanted to graft.")
        self.number_of_grafts_considered += len(this2anc) * len(other2anc)
        self.time_in_graft += time.time() - s

        self.this_number_of_grafts_considered = len(this2anc) * len(other2anc)
        self.this_time_in_graft += time.time() - s

    def updated_from_children(self, i):
        s = time.time()
        if len(self.children[i]) >= 2:
            self.num_descendants[i] = (
                self.num_descendants[self.children[i][0]]
                + self.num_descendants[self.children[i][1]]
            )
        self.scores[i] = -float("inf")
        self.needs_update_model[i] = True
        self.needs_update_desc[i] = True
        self.time_in_update += time.time() - s
        self.time_in_update_from_children += time.time() - s

    def update(self, i, use_tqdm=False):
        s = time.time()
        needs_update = []
        to_check = [i]
        while to_check:
            curr = to_check.pop(0)
            if self.needs_update_model[curr]:
                needs_update.append(curr)
                for c in self.get_children(curr):
                    to_check.append(c)
        self.time_in_update_walk += time.time() - s
        if use_tqdm:
            for j in range(len(needs_update) - 1, -1, -1):
                self.single_update(needs_update[j])
        else:
            for j in range(len(needs_update) - 1, -1, -1):
                self.single_update(needs_update[j])
        self.time_in_update += time.time() - s
        return needs_update

    def single_update(self, i):
        # print("updating node %s", i)
        assert self.needs_update_model[i]
        kids = self.get_children(i)
        s = time.time()
        c1 = self.get_sum(kids[0])
        c2 = self.get_sum(kids[1])
        self.num_descendants[i] = (
            self.num_descendants[kids[0]] + self.num_descendants[kids[1]]
        )
        self.set_sum(i, c1 + c2)
        self.compute_centroid(i)
        self.time_in_centroid += time.time() - s
        self.new_node[i] = False
        self.needs_update_model[i] = False

    def node_from_nodes(self, n1, n2):
        # print("creating new node from nodes %s and %s", n1, n2)
        new_node_id = self.next_node_id
        # print("new node is %s", new_node_id)
        self.grow_if_necessary()
        assert self.next_node_id < self.max_nodes
        assert self.next_node_id >= self.max_num_points
        self.next_node_id += 1
        self.needs_update_model[new_node_id] = True
        self.needs_update_desc[new_node_id] = True
        self.num_descendants[new_node_id] = (
            self.num_descendants[n1] + self.num_descendants[n2]
        )
        return new_node_id

    def grow_if_necessary(self):
        if self.next_node_id >= self.max_nodes:
            # print("resizing internal structures...")
            new_max_nodes = 2 * self.max_nodes
            # print("new max nodes %s", new_max_nodes)
            self.grow_nodes(new_max_nodes)
            self.grow_centroids_and_sums(new_max_nodes)
            self.max_nodes = new_max_nodes

    def grow_points_if_necessary(self):
        if self.point_counter >= self.max_num_points:
            new_max_num_points = 2 * self.max_num_points
            # print("new num points %s" % new_max_num_points)
            offset = new_max_num_points - self.max_num_points
            # grow internals if necessary
            if offset + self.next_node_id >= self.max_nodes:
                new_max_nodes = 2 * self.max_nodes
                # print("new max nodes %s", new_max_nodes)
                self.grow_nodes(new_max_nodes)
                self.grow_centroids_and_sums(new_max_nodes)
                self.max_nodes = new_max_nodes
            # re-map the internals
            self.remap_node(offset)
            self.remap_centroids_and_sums(offset)
            self.max_num_points = new_max_num_points
            self.next_node_id = self.next_node_id + offset

    def remap_node(self, parent_offset):
        for i in range(self.max_nodes - 1, self.max_num_points + parent_offset - 1, -1):
            self.children[i] = [
                c if c <= self.point_counter else c + parent_offset
                for c in self.children[i - parent_offset]
            ]
            self.descendants[i] = self.descendants[i - parent_offset]
            self.scores[i] = self.scores[i - parent_offset]
            self.needs_update_model[i] = self.needs_update_model[i - parent_offset]
            self.new_node[i] = self.new_node[i - parent_offset]
            self.needs_update_desc[i] = self.needs_update_desc[i - parent_offset]
            self.parent[i] = (
                self.parent[i - parent_offset] + parent_offset
                if self.parent[i - parent_offset] >= 0
                else self.parent[i - parent_offset]
            )
            self.num_descendants[i] = self.num_descendants[i - parent_offset]

        for i in range(0, self.point_counter, 1):
            self.parent[i] = (
                self.parent[i] + parent_offset
                if self.parent[i] >= 0
                else self.parent[i]
            )

        for i in range(
            self.max_num_points + parent_offset - 1, self.point_counter - 1, -1
        ):
            self.children[i].clear()
            self.descendants[i].clear()
            self.scores[i] = -np.Inf
            self.new_node[i] = False
            self.needs_update_desc[i] = False
            self.needs_update_model[i] = False
            self.parent[i] = -1
            self.num_descendants[i] = 1

    def remap_centroids_and_sums(self, parent_offset):
        for i in range(self.max_nodes - 1, self.max_num_points + parent_offset - 1, -1):
            self.centroids[i] = self.centroids[i - parent_offset]
            self.sums[i] = self.centroids[i - parent_offset]

        for i in range(
            self.max_num_points + parent_offset - 1, self.point_counter - 1, -1
        ):
            self.centroids[i] *= 0
            self.sums[i] *= 0

    def grow_nodes(self, new_max_nodes):
        self.sibs = None
        self.children.extend([[] for _ in range(new_max_nodes - self.max_nodes)])
        self.descendants.extend([[] for _ in range(new_max_nodes - self.max_nodes)])
        self.scores = np.hstack(
            [
                self.scores,
                -np.inf * np.ones(new_max_nodes - self.max_nodes, dtype=np.float32),
            ]
        )
        self.needs_update_model = np.hstack(
            [
                self.needs_update_model,
                np.zeros(new_max_nodes - self.max_nodes, dtype=np.bool_),
            ]
        )
        self.new_node = np.hstack(
            [self.new_node, np.ones(new_max_nodes - self.max_nodes, dtype=np.bool_)]
        )
        self.needs_update_desc = np.hstack(
            [
                self.needs_update_desc,
                np.zeros(new_max_nodes - self.max_nodes, dtype=np.bool_),
            ]
        )
        self.parent = np.hstack(
            [self.parent, -1 * np.ones(new_max_nodes - self.max_nodes, dtype=np.int32)]
        )
        self.num_descendants = np.hstack(
            [
                self.num_descendants,
                -1 * np.ones(new_max_nodes - self.max_nodes, dtype=np.float32),
            ]
        )

    def grow_centroids_and_sums(self, new_max_nodes):
        self.centroids = np.vstack(
            [
                self.centroids,
                np.zeros((new_max_nodes - self.max_nodes, self.dim_), dtype=np.float32),
            ]
        )
        self.sums = np.vstack(
            [
                self.sums,
                np.zeros((new_max_nodes - self.max_nodes, self.dim_), dtype=np.float32),
            ]
        )

    def add_pt(self, i):
        self.sums[i] += self.points_[i]
        self.num_descendants[i] = 1
        self.point_counter += 1
        self.descendants[i].append(i)
        self.compute_centroid(i)
        self.new_node[i] = False

    def build_dendrogram(self):
        # s = time.time()
        for i in range(self.num_points_):
            if i % 100 == 0:
                self.stats_string()
            self.insert(i)

    def insert(self, i, i_vec=None):
        if i_vec is not None:
            self.points_[i] = i_vec
        # s = time.time()
        # print("[insert] insert(%s)", i)
        # first point
        if self.point_counter == 0:
            self.add_pt(i)
        else:
            self.grow_points_if_necessary()
            i_vec = np.expand_dims(self.points_[i], 0)
            # dists, nns = self.cknn(i_vec, self.k, None, None)
            _, nns = self.cknn(i_vec, None, None)
            self.add_pt(i)
            sib = self.find_rotate(i, nns[0])
            parent = self.node_from_nodes(sib, i)
            self.make_sibling(sib, i, parent)
            curr_update = parent
            while curr_update != -1:
                self.updated_from_children(curr_update)
                curr_update = self.get_parent(curr_update)
            self.graft(parent)
        # t = time.time()
        # print("[insert] finished insert(%s) in %s seconds", i, t - s)

    def init_csim(self, sim):
        if sim == "dot":
            # print("Using csim = dot")
            self.csim = self.csim_dot
        elif sim == "l2":
            # print("Using csim = l2")
            self.csim = self.csim_l2
        elif sim == "sql2":
            # print("Using csim = sql2")
            self.csim = self.csim_sql2
        elif sim == "ward":
            # print("Using csim = sql2")
            self.csim = self.csim_ward

    def csim_dot(self, x, y):
        sims = np.matmul(x, y.transpose(1, 0))
        return sims

    def csim_l2(self, x, y):
        dists = cdist(x, y)
        return 1.0 / (1 + dists)

    def csim_sql2(self, x, y):
        dists = cdist(x, y, "sqeuclidean")
        return 1.0 / (1 + dists)

    def csim_ward(self, x, y, ndx, ndy):
        dists = (ndx * ndy)(ndx + ndy) * cdist(x, y, "sqeuclidean")
        return 1.0 / (1 + dists)

    def cknn(self, i_vec, offlimits1, offlimits2, pc=None):
        # k = min(self.point_counter, k)
        s = time.time()
        # TODO: Allow offlimits to 2D
        # import pdb; pdb.set_trace()
        if pc is None:
            pc = self.point_counter
        point_vecs = self.points_[0:pc]
        # sims = np.matmul(i_vec, point_vecs.transpose(1, 0))
        sims = self.csim(i_vec, point_vecs)
        if offlimits1 is not None:
            sims[:, offlimits1] = -float("Inf")
        if offlimits2 is not None and len(offlimits2) > 0:
            sims[:, offlimits2] = -float("Inf")
        indices = np.argmax(sims, axis=1)
        distances = sims[0, indices]
        indices = indices[distances != -np.Inf]
        distances = distances[distances != -np.Inf]
        self.time_in_search += time.time() - s
        return distances, indices

    def e_score_batch(self, i, j):
        if np.any(self.needs_update_model[i]):
            for ii in i:
                if self.needs_update_model[ii]:
                    self.update(ii)
        if np.any(self.needs_update_model[j]):
            for jj in j:
                if self.needs_update_model[jj]:
                    self.update(jj)

        i_vec = self.get_centroid_batch(i)
        j_vec = self.get_centroid_batch(j)

        s1 = time.time()
        # sims = np.matmul(i_vec, j_vec.transpose(1, 0))
        sims = self.csim(i_vec, j_vec)
        self.time_in_graft_score_only += time.time() - s1
        return sims

    def e_score(self, i, j):
        if self.needs_update_model[i]:
            self.update(i)

        if self.needs_update_model[j]:
            self.update(j)

        i_vec = self.get_centroid(i)
        j_vec = self.get_centroid(j)
        # sims = np.matmul(i_vec, j_vec.transpose(1, 0))
        sims = self.csim(i_vec, j_vec)

        return sims[0][0]

    def get_score_batch(self, i):
        # TODO: vectorize
        s = time.time()
        if not np.all(np.isfinite(self.scores[i])):
            for ii in i:
                if not np.isfinite(self.scores[ii]):
                    kids = self.get_children(ii)
                    res = self.e_score(kids[0], kids[1])
                    self.scores[ii] = res
        self.time_in_graft_get_scores += time.time() - s
        self.this_time_in_graft_get_scores += time.time() - s
        return self.scores[i]

    def get_score(self, i):
        """Get the linkage score at node with index i."""
        if not np.all(np.isfinite(self.scores[i])):
            kids = self.get_children(i)
            res = self.e_score(kids[0], kids[1])
            self.scores[i] = res
        return self.scores[i]

    def fix_scores(self):
        to_fix = np.where(np.isinf(self.scores))[0]
        for i in to_fix:
            kids = self.get_children(i)
            if len(kids) >= 2:
                res = self.e_score(kids[0], kids[1])
                self.scores[i] = res
        return self.scores

    def make_sibling(self, node, new_sib, parent):
        # print("make_sibling(node=%s, new_sib=%s, parent=%s)", node, new_sib, parent)
        sib_parent = self.get_parent(new_sib)
        # print(
        #     "make_sibling(node=%s, new_sib=%s, parent=%s) sib_parent=%s",
        #     node,
        #     new_sib,
        #     parent,
        #     sib_parent,
        # )
        if sib_parent != -1:
            sib_gp = self.get_parent(sib_parent)
            old_sib = self.get_sibling(new_sib)
            self.set_parent(old_sib, sib_gp)
            if sib_gp != -1:
                self.remove_child(sib_gp, sib_parent)
                self.add_child(sib_gp, old_sib)
            self.clear_children(sib_parent)
            self.parent[sib_parent] = -2  # Code for deletion
        else:
            assert self.is_leaf(new_sib), "self.is_leaf(new_sib=%s)=%s" % (
                new_sib,
                self.is_leaf(new_sib),
            )

        self.set_parent(parent, self.get_parent(node))
        parentparent = self.get_parent(parent)
        if parentparent != -1:
            self.remove_child(parentparent, node)
            self.add_child(parentparent, parent)
        self.add_child(parent, node)
        self.add_child(parent, new_sib)
        self.set_parent(node, parent)
        self.set_parent(new_sib, parent)

    def is_leaf(self, i):
        return len(self.get_children(i)) == 0

    def get_parent(self, i):
        return self.parent[i]

    def set_parent(self, i, p):
        self.parent[i] = p

    def get_sibling(self, i):
        p = self.get_parent(i)
        return [x for x in self.get_children(p) if x != i][0]

    def get_ancs_with_self(self, i):
        needs_anc = [i]
        # walk up until we find someone who has known ancestors
        curr = self.get_parent(i)
        while curr != -1:
            needs_anc.append(curr)
            curr = self.get_parent(curr)
        return needs_anc

    def get_ancs(self, i):
        # TODO speed up w/ cache.
        needs_anc = []
        # walk up until we find someone who has known ancestors
        curr = self.get_parent(i)
        while curr != -1:
            needs_anc.append(curr)
            curr = self.get_parent(curr)
        return needs_anc

    def get_children(self, i):
        return self.children[i]

    def add_child(self, p, c):
        self.children[p].append(c)

    def remove_child(self, p, c):
        # assert c in self.children[p], "trying to remove c=%s from p=%s with kids=%s" % (
        #     c,
        #     p,
        #     str(self.children[p]),
        # )
        if c in self.children[p]:
            self.children[p].remove(c)

    def clear_children(self, i):
        self.children[i].clear()

    def write_tree(self, filename, lbls):
        # print("writing tree to file %s", filename)
        with open(filename, "w") as fin:
            for i in range(self.num_points_):
                fin.write("%s\t%s\t%s\n" % (i, self.get_parent(i), lbls[i]))
            for j in range(self.num_points_, self.next_node_id):
                if self.parent[j] != -2:
                    fin.write("%s\t%s\tNone\n" % (j, self.parent[j]))
            r = self.root()
            fin.write("-1\tNone\tNone\n" % r)

    def root(self):
        r = 0
        while self.get_parent(r) != -1:
            r = self.get_parent(r)
        return r

    def ancestors(self, i):
        r = i
        ancs = [r]
        while self.get_parent(r) != -1:
            r = self.get_parent(r)
            ancs.append(r)
        return ancs

    def flat_clustering(self, threshold):
        frontier = [self.root()]
        clusters = []
        while frontier:
            n = frontier.pop(0)
            if len(self.children[n]) != 0 and self.get_score(n) < threshold:
                frontier.extend(self.children[n])
            else:
                clusters.append(n)
        assignments = -1 * np.ones(self.num_points_, np.int32)
        for c_idx, c in enumerate(clusters):
            for d in self.get_descendants(c):
                assignments[d] = c_idx
        return assignments

    def k_clustering(self, k):
        scores = self.fix_scores()
        clusters = [self.root()]
        anomalies = []
        while len(clusters) < k and len(clusters) > 0:
            i = clusters.pop(np.argmin(scores[clusters]))
            children = self.get_children(i)
            for j in children:
                if self.is_leaf(j):
                    anomalies.append(j)
                else:
                    clusters.append(j)

        assignments = -1 * np.ones(self.num_points_, np.int32)
        for c_idx, c in enumerate(clusters):
            c_points = self.get_descendants(c)
            # print(c_points)
            assignments[c_points] = c_idx
        # print(np.unique(assignments))
        # for i in range(k):
        #     if not np.any(assignments == i) and not np.all(assignments == -1):
        #         print(k)
        return assignments


if __name__ == "__main__":
    n = 200
    point_labels = np.random.random_integers(0, 10, n)
    vectors = np.random.random((n, 5)).astype(np.float32)
    grinch = Grinch(n_clusters=5)
    grinch.fit(vectors)
    vectors = np.random.random((n, 5)).astype(np.float32)
    grinch.partial_fit(vectors)
    # grinch.write_tree("tmp/tree.out", np.random.random_integers(0, 10, n * 2))
    y = grinch.predict(vectors)
    # y = grinch.flat_clustering(0.88)
    print(np.unique(y))
    print(np.unique(grinch.labels_))

