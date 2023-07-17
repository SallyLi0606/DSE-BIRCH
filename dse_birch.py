import warnings
import numpy as np
from sklearn.utils import check_array
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms, safe_sparse_dot
from sklearn.cluster import AgglomerativeClustering
warnings.filterwarnings(action='ignore')

def _split_node(node, threshold1, threshold2, branching_factor, alpha=0.1):
    """ Split the current node into two subclusters.
        The node has to be split if there is no place for a new subcluster in the node.
        1. Two empty nodes and two empty subclusters are initialized.
        2. The pair of distant subclusters are found.
        3. The properties of the empty subclusters and nodes are updated
        according to the nearest distance between the subclusters to the
        pair of distant subclusters.
        4. The two nodes are set as children to the two subclusters.
    """
    new_subcluster1 = _CFSubcluster()
    new_subcluster2 = _CFSubcluster()
    new_node1 = _CFNode(
        threshold1, threshold2, branching_factor,
        is_leaf=node.is_leaf, n_features=node.n_features, alpha=alpha)
    new_node2 = _CFNode(
        threshold1, threshold2, branching_factor,
        is_leaf=node.is_leaf, n_features=node.n_features, alpha=alpha)
    new_subcluster1.child_ = new_node1
    new_subcluster2.child_ = new_node2

    if node.is_leaf:
        if node.prev_leaf_ is not None:
            node.prev_leaf_.next_leaf_ = new_node1
        new_node1.prev_leaf_ = node.prev_leaf_
        new_node1.next_leaf_ = new_node2
        new_node2.prev_leaf_ = new_node1
        new_node2.next_leaf_ = node.next_leaf_
        if node.next_leaf_ is not None:
            node.next_leaf_.prev_leaf_ = new_node2

    dist1 = euclidean_distances(
        node.centroids_[:, 0, :], Y_norm_squared=node.squared_norm_[:, 0], squared=True)
    dist2 = euclidean_distances(
        node.centroids_[:, 1, :], Y_norm_squared=node.squared_norm_[:, 1], squared=True)
    n_clusters = dist1.shape[0]
    dist = dist1 + alpha * dist2

    farthest_idx = np.unravel_index(dist.argmax(), (n_clusters, n_clusters))
    node1_dist, node2_dist = dist[(farthest_idx,)]

    node1_closer = node1_dist < node2_dist
    # make sure node1 is closest to itself even if all distances are equal.
    # This can only happen when all node.centroids_ are duplicates leading to all
    # distances between centroids being zero.
    node1_closer[farthest_idx[0]] = True

    for idx, subcluster in enumerate(node.subclusters_):
        if node1_closer[idx]:
            new_node1.append_subcluster(subcluster)
            new_subcluster1.update(subcluster)
        else:
            new_node2.append_subcluster(subcluster)
            new_subcluster2.update(subcluster)
    return new_subcluster1, new_subcluster2


class _CFSubcluster(object):
    """ Each subcluster in a CFNode is called a CFSubcluster.
        A CFSubcluster can have a CFNode has its child.

    Parameters
    ----------
    linear_sum : ndarray of shape (n_features,), default=None
        Sample. This is kept optional to allow initialization of empty
        subclusters.

    Attributes
    ----------
    n_samples_ : int
        Number of samples that belong to each subcluster.

    linear_sum_ : ndarray
        Linear sum of all the samples in a subcluster. Prevents holding
        all sample data in memory.

    squared_sum_ : float
        Sum of the squared l2 norms of all samples belonging to a subcluster.

    centroid_ : ndarray of shape (branching_factor + 1, n_features)
        Centroid of the subcluster. Prevent recomputing of centroids when
        ``CFNode.centroids_`` is called.

    child_ : _CFNode
        Child Node of the subcluster. Once a given _CFNode is set as the child
        of the _CFNode, it is set to ``self.child_``.

    sq_norm_ : ndarray of shape (branching_factor + 1,)
        Squared norm of the subcluster. Used to prevent recomputing when
        pairwise minimum distances are computed.
    """
    def __init__(self, linear_sum1=None, linear_sum2=None):
        if linear_sum1 is None:
            self.n_samples_ = 0
            self.squared_sum1_ = 0.0
            self.linear_sum1_ = 0
            self.squared_sum2_ = 0.0
            self.linear_sum2_ = 0

        elif linear_sum1 is not None and linear_sum2 is not None:
            self.n_samples_ = 1
            self.centroid1_ = self.linear_sum1_ = linear_sum1
            self.squared_sum1_ = self.sq_norm1_ = np.dot(self.linear_sum1_, self.linear_sum1_)
            self.centroid2_ = self.linear_sum2_ = linear_sum2
            self.squared_sum2_ = self.sq_norm2_ = np.dot(self.linear_sum2_, self.linear_sum2_)
            
            self.centroid_ = np.stack([self.centroid1_, self.centroid2_], axis=0)
            self.sq_norm_ = np.array([np.dot(self.centroid1_, self.centroid1_),
                                      np.dot(self.centroid2_, self.centroid2_)])

        else:
            raise ValueError("Both linear_sum1 and linear_sum2 should be either all empty or all non-empty.")
        self.child_ = None

    def update(self, subcluster):
        self.n_samples_ += subcluster.n_samples_
        self.linear_sum1_ += subcluster.linear_sum1_
        self.squared_sum1_ += subcluster.squared_sum1_
        self.centroid1_ = self.linear_sum1_ / self.n_samples_
        self.sq_norm1_ = np.dot(self.centroid1_, self.centroid1_)
        self.linear_sum2_ += subcluster.linear_sum2_
        self.squared_sum2_ += subcluster.squared_sum2_
        self.centroid2_ = self.linear_sum2_ / self.n_samples_
        self.sq_norm2_ = np.dot(self.centroid2_, self.centroid2_)

        self.centroid_ = np.stack([self.centroid1_, self.centroid2_], axis=0)
        self.sq_norm_ = np.array([self.sq_norm1_, self.sq_norm2_])

    def merge_subcluster(self, nominee_cluster, threshold1, threshold2):
        """
        Calculate the radius of the new cluster and compare it with the threshold.
        Check if merging is possible. If the condition is met, merge the clusters.
        The threshold is the specified value. The new cluster will be merged if its radius is smaller than the threshold; otherwise, it will be split.
        Note: Only the radius is considered when determining if merging is possible,
                and the impact of the new subcluster on exceeding the capacity is not taken into account.
        """
        new_n = self.n_samples_ + nominee_cluster.n_samples_

        new_ss1 = self.squared_sum1_ + nominee_cluster.squared_sum1_
        new_ls1 = self.linear_sum1_ + nominee_cluster.linear_sum1_
        new_centroid1 = (1 / new_n) * new_ls1
        new_norm1 = np.dot(new_centroid1, new_centroid1)
        # The squared radius of the cluster is defined:
        #   r^2  = sum_i ||x_i - c||^2 / n
        # with x_i the n points assigned to the cluster and c its centroid:
        #   c = sum_i x_i / n
        # This can be expanded to:
        #   r^2 = sum_i ||x_i||^2 / n - 2 <sum_i x_i / n, c> + n ||c||^2 / n
        # and therefore simplifies to:
        #   r^2 = sum_i ||x_i||^2 / n - ||c||^2
        sq_radius1 = new_ss1 / new_n - new_norm1

        new_ss2 = self.squared_sum2_ + nominee_cluster.squared_sum2_
        new_ls2 = self.linear_sum2_ + nominee_cluster.linear_sum2_
        new_centroid2 = (1 / new_n) * new_ls2
        new_norm2 = np.dot(new_centroid2, new_centroid2)
        sq_radius2 = new_ss2 / new_n - new_norm2

        new_centroid = np.stack([new_centroid1, new_centroid2], axis=0)
        new_norm = np.array([new_norm1, new_norm2])

        # If the radius of the merged cluster is smaller than the threshold, merge the clusters and update the CF of the merged cluster.
        if sq_radius1 <= threshold1**2 and sq_radius2 <= threshold2**2:
            (self.n_samples_, self.linear_sum1_, self.squared_sum1_, self.centroid1_,
             self.sq_norm1_, self.linear_sum2_, self.squared_sum2_, self.centroid2_,
             self.sq_norm2_, self.centroid_, self.sq_norm_) = \
                                new_n, new_ls1, new_ss1, new_centroid1, new_norm1, \
                                new_ls2, new_ss2, new_centroid2, new_norm2, new_centroid, new_norm
            return True
        # otherwise, do not split
        return False


class _CFNode(object):
    """ Each node in a CFTree is called a CFNode.
        
        Attributes
        ----------
        subclusters_ : list
            List of subclusters for a particular CFNode.

        prev_leaf_ : _CFNode
            Useful only if is_leaf is True.

        next_leaf_ : _CFNode
            next_leaf. Useful only if is_leaf is True.
            the final subclusters.

        init_centroids_ : ndarray of shape (branching_factor + 1, n_features)
            Manipulate ``init_centroids_`` throughout rather than centroids_ since
            the centroids are just a view of the ``init_centroids_`` .

        init_sq_norm_ : ndarray of shape (branching_factor + 1,)
            manipulate init_sq_norm_ throughout. similar to ``init_centroids_``.

        centroids_ : ndarray of shape (branching_factor + 1, n_features)
            View of ``init_centroids_``.

        squared_norm_ : ndarray of shape (branching_factor + 1,)
            View of ``init_sq_norm_``.
    """
    def __init__(self, threshold1, threshold2, branching_factor, is_leaf, n_features, alpha=0.1):
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.branching_factor = branching_factor
        self.is_leaf = is_leaf
        self.n_features = n_features
        self.alpha = alpha

        self.subclusters_ = [] # a list of subclusters
        self.init_centroids_ = np.zeros((branching_factor+1, 2, n_features))
        self.init_sq_norm_ = np.zeros((branching_factor+1, 2))
        self.squared_norm_ = []
        self.prev_leaf_ = None
        self.next_leaf_ = None

    def append_subcluster(self, subcluster):
        n_samples = len(self.subclusters_)
        self.subclusters_.append(subcluster)
        self.init_centroids_[n_samples] = subcluster.centroid_
        self.init_sq_norm_[n_samples] = subcluster.sq_norm_
        # Keep centroids and squared norm as views. In this way
        # if we change init_centroids and init_sq_norm_, it is sufficient
        self.centroids_ = self.init_centroids_[:n_samples+1, :, :]
        self.squared_norm_ = self.init_sq_norm_[:n_samples+1, :]

    def update_split_subclusters(self, subcluster, new_subcluster1, new_subcluster2):
        # Remove one subcluster from a node and then add two new subclusters.
        ind = self.subclusters_.index(subcluster)
        self.subclusters_[ind] = new_subcluster1
        self.init_centroids_[ind] = new_subcluster1.centroid_
        self.init_sq_norm_[ind] = new_subcluster1.sq_norm_
        # Add `new_subcluster2` to the node, thus adding the two new subclusters to the node.
        self.append_subcluster(new_subcluster2)

    def insert_cf_subcluster(self, subcluster):
        """ Insert a new subcluster into the node and return a flag indicating whether to split the current node's subclusters.
        """
        if not self.subclusters_:
            self.append_subcluster(subcluster)
            return False

        threshold1 = self.threshold1
        threshold2 = self.threshold2
        branching_factor = self.branching_factor
        dist_matrix1 = np.dot(self.centroids_[:, 0, :], subcluster.centroid_[0, :])
        dist_matrix1 *= -2.0
        dist_matrix1 += self.squared_norm_[:, 0]
        dist_matrix2 = np.dot(self.centroids_[:, 1, :], subcluster.centroid_[1, :])
        dist_matrix2 *= -2.0
        dist_matrix2 += self.squared_norm_[:, 1]

        dist_matrix = dist_matrix1 + self.alpha * dist_matrix2

        closest_index = np.argmin(dist_matrix)
        closest_subcluster = self.subclusters_[closest_index]

        if closest_subcluster.child_ is not None:
            split_child = closest_subcluster.child_.insert_cf_subcluster(subcluster)
            if not split_child:
                closest_subcluster.update(subcluster)
                self.init_centroids_[closest_index] = self.subclusters_[closest_index].centroid_
                self.init_sq_norm_[closest_index] = self.subclusters_[closest_index].sq_norm_
                return False
            else:
                new_subcluster1, new_subcluster2 = _split_node(closest_subcluster.child_, threshold1, threshold2, branching_factor, self.alpha)
                self.update_split_subclusters(closest_subcluster, new_subcluster1, new_subcluster2)
                if len(self.subclusters_) > branching_factor:
                    return True
                return False

        else:
            merged = closest_subcluster.merge_subcluster(subcluster, threshold1, threshold2)
            if merged:
                self.init_centroids_[closest_index] = closest_subcluster.centroid_
                self.init_sq_norm_[closest_index] = closest_subcluster.sq_norm_
                return False
            elif len(self.subclusters_) < branching_factor:
                self.append_subcluster(subcluster)
                return False
            else:
                self.append_subcluster(subcluster)
                return True


class DSE_Birch():
    def __init__(self, threshold1=0.5, threshold2=0.5, branching_factor=50,
                 n_clusters=3, compute_labels=True, alpha=0.1):
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.compute_labels = compute_labels
        self.alpha = alpha

    def fit(self, X1, X2):
        """
        Build a CF Tree for the input data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        self
            Fitted estimator.
        
        Attributes
        ----------
        root_ : _CFNode
            Root of the CFTree.

        dummy_leaf_ : _CFNode
            Start pointer to all the leaves.
        """
        threshold1 = self.threshold1
        threshold2 = self.threshold2
        X1 = check_array(X1, accept_sparse='csr', copy=True) # static
        X2 = check_array(X2, accept_sparse='csr', copy=True) # dynamic
        branching_factor = self.branching_factor
        if branching_factor <= 1:
            raise ValueError("Branching_factor should be greater than one.")
        assert X1.shape == X2.shape, "Shape of input `X1` and `X2` should be the same!"
        n_samples, n_features = X1.shape
        
        self.root_ = _CFNode(threshold1, threshold2, branching_factor, is_leaf=True, n_features=n_features, alpha=self.alpha)
        self.dummy_leaf_ = _CFNode(threshold1, threshold2, branching_factor, is_leaf=True, n_features=n_features, alpha=self.alpha)
        self.dummy_leaf_.next_leaf_ = self.root_
        self.root_.prev_leaf_ = self.dummy_leaf_
        
        for sample1, sample2 in zip(X1, X2):
            subcluster = _CFSubcluster(linear_sum1=sample1, linear_sum2=sample2)
            split = self.root_.insert_cf_subcluster(subcluster)
            if split:
                new_subcluster1, new_subcluster2 = _split_node(self.root_, threshold1, threshold2, branching_factor, self.alpha)
                del self.root_
                self.root_ = _CFNode(threshold1, threshold2, branching_factor, is_leaf=False, n_features=n_features, alpha=self.alpha)
                self.root_.append_subcluster(new_subcluster1)
                self.root_.append_subcluster(new_subcluster2)

        centroids = np.concatenate([leaf.centroids_ for leaf in self._get_leaves()]) # shape: (num_samples, 2, n_feat)
        self.subcluster_centers_ = centroids

        self._global_clustering(X1, X2)
        return self

    def _get_leaves(self):
        """ Return the leaf nodes of CFNode.
        """
        leaf_ptr = self.dummy_leaf_.next_leaf_
        leaves = []
        while leaf_ptr is not None:
            leaves.append(leaf_ptr)
            leaf_ptr = leaf_ptr.next_leaf_
        return leaves

    def predict(self, X1, X2=None):
        """ To predict the cluster to which sample points in X belong using a fitted BIRCH model:
                select the cluster with the closest distance to the cluster center as the assigned cluster for each sample point.
        """
        reduced_distance1 = safe_sparse_dot(X1, self.subcluster_centers_[:, 0, :].T) # (num_test_samples, num_train_samples)
        reduced_distance1 *= -2
        reduced_distance1 += self._subcluster_norms1 # =row_norms(self.subcluster_centers_, squared=True)
        reduced_distance2 = safe_sparse_dot(X2, self.subcluster_centers_[:, 1, :].T)
        reduced_distance2 *= -2
        reduced_distance2 += self._subcluster_norms2

        reduced_distance = reduced_distance1 + self.alpha * reduced_distance2

        return self.subcluster_labels_[np.argmin(reduced_distance, axis=1)]

    def _global_clustering(self, X1=None, X2=None):
        """ To further operate on the clusters obtained from fitting the model with `fit`:
                evaluate the relationship between the number of clusters found by DSE-BIRCH and the specified `n_clusters`,
                and determine whether cluster aggregation is needed.
        """
        clusterer = self.n_clusters
        centroids = self.subcluster_centers_
        compute_labels = (X1 is not None) and self.compute_labels

        not_enough_centroids = False
        if isinstance(clusterer, int):
            clusterer = AgglomerativeClustering(n_clusters=self.n_clusters)
            if len(centroids) < self.n_clusters:
                not_enough_centroids = True
        elif (clusterer is not None):
            raise ValueError("n_clusters should be an instance of " "ClusterMixin or an int")

        self._subcluster_norms1 = row_norms(self.subcluster_centers_[:, 0, :], squared=True)
        self._subcluster_norms2 = row_norms(self.subcluster_centers_[:, 1, :], squared=True)

        if clusterer is None or not_enough_centroids:
            self.subcluster_labels_ = np.arange(len(centroids))
            if not_enough_centroids:
                warnings.warn(
                    "Number of subclusters found (%d) by Birch is less than (%d). Decrease the threshold." % (
                    len(centroids), self.n_clusters))
        else:
            tmp_subcluster_centers_ = self.subcluster_centers_.copy()
            tmp_subcluster_centers_[:, 1, :] *= self.alpha
            tmp_subcluster_centers_ = tmp_subcluster_centers_.reshape(tmp_subcluster_centers_.shape[0], -1)
            self.subcluster_labels_ = clusterer.fit_predict(tmp_subcluster_centers_)

        if compute_labels:
            self.labels_ = self.predict(X1, X2)
