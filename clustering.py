import numpy as np
from dse_birch import DSE_Birch
from utils import evaluate_metrics

import warnings
warnings.filterwarnings(action='ignore')


class BaseClass:
    """ Base class for clustering methods, used for data initialization and retrieval
    """
    def __init__(self, C_train, C_hat_train, Y_train, C_test, C_hat_test, Y_test, verbose=False):
        self.C_train = C_train
        self.C_hat_train = C_hat_train
        self.Y_train = Y_train
        self.C_test = C_test
        self.C_hat_test = C_hat_test
        self.Y_test = Y_test
        self.verbose = verbose

    def get_data(self, const_deriv='const'):
        """ Return constant or (and) derivative feature
        """
        if const_deriv == 'const':
            X_train = self.C_train
            X_test = self.C_test
            Y_test = self.Y_test
        elif const_deriv == 'deriv':
            X_train = self.C_hat_train
            X_test = self.C_hat_test
            Y_test = self.Y_test
        elif const_deriv == 'both':
            X_train = np.concatenate((self.C_train, self.C_hat_train), axis=1)
            X_test = np.concatenate((self.C_test, self.C_hat_test), axis=1)
            Y_test = self.Y_test
        else:
            raise ValueError()
        return  X_train, X_test, Y_test
    

class DSEBirchClustering(BaseClass):
    """ Our proposed DSE-BIRCH clustering algorithm
    """
    def __init__(self, C_train, C_hat_train, Y_train, C_test, C_hat_test, Y_test, verbose=False, return_metrics=True):
        super().__init__(C_train, C_hat_train, Y_train, C_test, C_hat_test, Y_test, verbose)
        self.return_metrics = return_metrics

    def Birch_proposed(self, branching_factor=50, n_clusters=2, threshold1=0.5, threshold2=0.5, _lambda=0.1):
        birch = DSE_Birch(threshold1=threshold1, threshold2=threshold2, branching_factor=branching_factor,
                      n_clusters=n_clusters, alpha=_lambda)
        birch.fit(self.C_train, self.C_hat_train)
        cluster_labels = birch.predict(self.C_test, self.C_hat_test)

        if self.return_metrics:
            result_metrics =evaluate_metrics(
                self.C_test, self.C_hat_test, self.Y_test, cluster_labels, birch.subcluster_centers_[cluster_labels], verbose=self.verbose, const_deriv='proposed')
            return result_metrics
        else:
            return None
        