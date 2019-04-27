import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import kmeanspp
from sklearn.metrics import pairwise_distances

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Kmeansll:
    """K-Meansll Clustering Algorithm"""
    
    def __init__(self, k, omega, centers=None, cost=None,iter=None, labels=None, max_iter = 1000):
        """Initialize Parameters"""
        
        self.max_iter = max_iter
        self.k = k
        self.omega = omega
        self.centers = np.empty(1)
        self.cost = []
        self.iter = 1
        self.labels = np.empty(1)
        
    def calc_weight(self, data, centers):
        """Weight Calculation"""
        
        l = len(centers)
        distance = pairwise_distances(data, centers)
        labels = np.argmin(distance, axis=1)
        weights = [sum(labels == i) for i in range(l)]
        return (weights/sum(weights))

    def calc_distances(self, data, centers, weights):
        """Distance Matrix"""
        
        distance = pairwise_distances(data, centers)**2
        min_distance = np.min(distance, axis = 1)
        D = min_distance*weights
        return D
    
    def initial_centers_Kmeansll(self, data, k, omega, weights):    
        """Initialize Centers for K-Meansll"""
        
        centers = []
        centers.append(random.choice(data))
        phi = np.int(np.round(np.log(sum(self.calc_distances(data, centers, weights)))))
        l = k*omega ## oversampling factor
        for i in range(phi):
            dist = self.calc_distances(data, centers, weights)
            prob = l*dist/sum(dist)
            for i in range(len(prob)):
                if prob[i] > np.random.uniform():
                    centers.append(data[i])
        centers = np.array(centers)
        recluster_weight = self.calc_weight(data, centers)
        reclusters = kmeanspp.Kmeanspp(k).fit(centers, recluster_weight).labels
        initial_centers = []
        for i in np.unique(reclusters):
            initial_centers.append(np.mean(centers[reclusters == i], axis = 0))
        return initial_centers
    
    
    def fit(self, data, weights=None):
        """Clustering Process"""
        
        if weights is None: weights = np.ones(len(data))
        if type(data) == pd.DataFrame: data=data.values
        nrow = data.shape[0]
        self.centers = self.initial_centers_Kmeansll(data, self.k, self.omega, weights)
        while (self.iter <= self.max_iter):
            distance = pairwise_distances(data, self.centers)**2
            self.cost.append(sum(np.min(distance, axis=1)))
            self.labels = np.argmin(distance, axis=1)
            centers_new = np.array([np.mean(data[self.labels == i], axis=0) for i in np.unique(self.labels)])
            
            ## sanity check
            if(np.all(self.centers == centers_new)): break 
            self.centers = centers_new
            self.iter += 1
        
        ## convergence check
        if (sum(np.min(pairwise_distances(data, self.centers)**2, axis=1)) != self.cost[-1]):
            warnings.warn("Algorithm Did Not Converge In {} Iterations".format(self.max_iter))
        return self

