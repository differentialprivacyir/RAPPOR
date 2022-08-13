import mmh3
import numpy as np
from sklearn.linear_model import ElasticNet


# class for creating bloom filter's hash
class Hash:

    def __init__(self, k, m, h):
        self.k = k  # number of bloom filter bits
        self.m = m  # number of cohorts
        self.h = h  # number of hash functions

    def hash_functions(self):
        hash_functions = []
        for cohort in range(0, self.m):
            hash_functions.append([lambda data: mmh3.hash(data, cohort+i) % self.k for i in range(0, self.h)])
        return hash_functions


class Server:

    def __init__(self, f, p, q, k, m, h, alpha):
        self.f = f  # privacy parameter for prr
        self.p = p  # privacy parameter for irr
        self.q = q  # privacy parameter for irr
        self.k = k  # number of bloom filter bits
        self.m = m  # number of cohorts
        H = Hash(k, m, h)
        self.hash_functions = H.hash_functions()  # bloom filter's hash functions
        self.reports = [np.zeros(self.k) for i in range(0, self.m)]  # clients reports (zeroed bloom filters of size k)
        self.cohorts = np.zeros(self.m)  # number of reports of each cohort
        self.alpha = alpha  # estimation parameter
        self.estimated = dict()  # estimation of number of times each Word reported

    def collect(self, data):  # collecting clients reports
        report, cohort = data  # each data consists of client's cohort number and privatized bloom filter
        self.cohorts[cohort] += 1
        self.reports[cohort] += report

    def _matrix_x(self, data):
        """
        Create a design matrix X of size km × M where M is the number of
        candidate strings under consideration. X is mostly 0 (sparse)
        with 1’s at the Bloom filter bits for each string for each cohort.
        """
        x = np.empty((self.k * len(self.cohorts), len(data)))
        for i, datum in enumerate(data):
            col = np.zeros((len(self.cohorts), self.k))
            for index, funcs in enumerate(self.hash_functions):
                for func in funcs:
                    col[index][func(datum)] = 1
            x[:, i] = col.flatten()
        return x

    def _vector_y(self):
        """
        Y is a vector of tij's, i ∈ [1, k], j ∈ [1, m]
        tij is estimated number of times each bit i within cohort j  is truly set in B for each cohort.
        tij = [cij − (p + 0.5 f q − 0.5 f p) Nj] / [(1 − f)(q − p)]
        Nj is number of reports in cohort j
        """
        y = np.array([])
        for i, report in enumerate(self.reports):
            scaled_bloom = (report - (self.p + 0.5 * self.f * (self.q - self.p)) * self.cohorts[i]) / \
                           ((1 - self.f) * (self.q - self.p))
            y = np.concatenate((y, scaled_bloom))
        return y

    def estimation(self, data):  # estimating number of times each Word reported. data is all the Words
        x = self._matrix_x(data)
        y = self._vector_y()

        # Using Lasso model and ElasticNet to fit a model Y ∼ X
        model = ElasticNet(positive=True, alpha=self.alpha, l1_ratio=0.4, fit_intercept=False, max_iter=10000)
        model.fit(x, y)
        estimated = model.coef_ * self.m

        for index, datum in enumerate(data):
            self.estimated[datum] = estimated[index]
        return self.estimated
