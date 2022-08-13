import json
import mmh3
import numpy as np


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


class Client:

    def __init__(self, f, p, q, k, m, h):
        self.f = f  # privacy parameter for prr
        self.p = p  # privacy parameter for irr
        self.q = q  # privacy parameter for irr
        self.k = k  # number of bloom filter bits
        self.cohort = np.random.randint(0, m)  # select client's cohort number randomly from 0 to m
        H = Hash(k, m, h)
        self.hash_functions = H.hash_functions()  # bloom filter's hash functions
        self.prr_cache = dict()   # memoizing cache for

    def report(self, data): # function for reporting the client's data to the server
        report = np.zeros(self.k)   # create the bloom filter string with all bit zero
        for func in self.hash_functions[self.cohort]:  # fill the bloom filter
            report[func(str(data))] = 1
        prr = self._prr(report)   # apply prr on the report
        irr = self._irr(prr)   # apply irr on the prr's output
        return irr, self.cohort

    def _prr(self, data):

        # if there is a memoized value for this data, exit the function
        cache_key = json.dumps(data.tolist())
        if self.prr_cache.get(cache_key) is not None:
            return self.prr_cache[cache_key]

        # apply prr on the data
        for index, bit in enumerate(data):
            rand = np.random.random()
            if rand < (0.5 * self.f):
                data[index] = 1
            elif rand < self.f:
                data[index] = 0

        self.prr_cache[cache_key] = data   # memoize the prr result
        return data

    def _irr(self, data):
        # apply irr on the data
        for index, bit in enumerate(data):
            rand = np.random.random()
            if bit == 1:
                data[index] = rand < self.q
            else:
                data[index] = rand < self.p
        return data
