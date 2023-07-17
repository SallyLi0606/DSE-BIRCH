import numpy as np
from sklearn import metrics
import numpy as np


def purity_score(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)

def evaluate_metrics(C_test, C_hat_test, Y_test, cluster_labels, cluster_centroids, verbose=False, const_deriv='const'):
    """ Evaluation of multiple metrics for clustering performance
        Args:
            C_test: Coefficient feature C of the test set
            Y_test: True labels of the test set
            cluster_labels: Cluster labels
            cluster_centroids: Coordinates of the cluster centroids
            verbose: Whether to print metric values
    """
    X = np.concatenate([C_test, C_hat_test], axis=1)

    silhouette_coefficient = metrics.silhouette_score(X, cluster_labels, metric='euclidean')
    DB_index = metrics.davies_bouldin_score(X, cluster_labels)
    purity = purity_score(Y_test, cluster_labels)
    RI = metrics.rand_score(Y_test, cluster_labels)
    NMI = metrics.normalized_mutual_info_score(Y_test, cluster_labels)

    if verbose:
        print(f'Silhouette Coefficient: {silhouette_coefficient:.4f}')
        print(f'Davies-Bouldin Index: {DB_index:.4f}')
        print(f'Purity: {purity:.4f}')
        print(f'Rand Index: {RI:.4f}')
        print(f'Normalized Mutual Information (NMI): {NMI:.4f}')

    result_metric = {
        'silhouette_coefficient': silhouette_coefficient,
        'DB_index': DB_index,
        'purity': purity,
        'RI': RI,
        'NMI': NMI
    }

    return result_metric
