import numpy as np
# import scipy.spatial.distance as sd
from sklearn.neighbors import NearestNeighbors
import math

class Sp:

    def __init__(self, data, sample_size, rseed=0):
        self.sample_size = sample_size
        self.ndata = len(data)
        self.ndim = len(data[0])
        # get samples
        np.random.seed(rseed)
        d_ids = np.array([i for i in range(self.ndata)])
        np.random.shuffle(d_ids)
        sample_ids = d_ids[0:self.sample_size]
        self.samples = data[sample_ids,:]
        # neigbourhood search model       
        self.neighourhood = NearestNeighbors(n_neighbors=1)
        self.neighourhood.fit(self.samples)
 

    def get_anomaly_score(self, x):
        X = np.reshape(x, (1,-1))
        dist, indices = self.neighourhood.kneighbors(X, n_neighbors=1, return_distance=True)
        return dist[0,0]            
