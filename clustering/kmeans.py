"""
    This is a file you will have to fill in.

    It contains helper functions required by K-means method via iterative improvement

"""
import numpy as np
import copy
import random

def init_centroids(k, inputs):
    """
    Selects k random rows from inputs and returns them as the chosen centroids

    :param k: number of cluster centroids, an int
    :param inputs: a 2D Python list, each row of which is one input
    """

    k_samples = random.sample(inputs, k)
    center = np.array(k_samples)
    return center

def assign_step(inputs, centroids):
    """
    Determines a centroid index for every row of the inputs using Euclidean Distance

    :param inputs: inputs of data, a 2D Python list
    :param centroids: a Numpy array of k current centroids
    """

    indice = [0.0] * len(inputs)
    for i in range(len(inputs)):
        m = float('inf')
        cen_ls = []
        for k in range(len(centroids)):
            if np.linalg.norm(inputs[i]-centroids[k]) < m:
                m = np.linalg.norm(inputs[i]-centroids[k])
                cen_ls.append(k)
                n = max(cen_ls)
        indice[i] = copy.deepcopy(n)
    return indice
def update_step(inputs, indices, k):
    """
    Computes the centroid for each cluster - the average of all data points in the cluster

    :param inputs: inputs of data, a 2D Python list
    :param indices: a Python list of centroid indices, one for each row of the inputs
    :param k: number of cluster centroids, an int
    """

    inputs = np.array(inputs)
    indices = np.array(indices)
    cen = list(set(indices))
    store = [[]] * k
    for k in range(k):
        store[k] = np.mean(inputs[indices==cen[k]],axis=0)
    return np.array(store)
def kmeans(inputs, k, max_iter, tol):
    """
    Runs the K-means algorithm on n rows of inputs using k clusters via iterative improvement
    Use init_centroids, assign_step, and update_step!
    The only computation that should occur within this function is checking 
    for convergence - everything else should be handled by helpers

    :param inputs: inputs of data, a 2D Python list
    :param k: number of cluster centroids, an int
    :param max_iter: the maximum number of times the algorithm can iterate trying to optimize the centroid values, an int
    :param tol: relative tolerance with regards to inertia to declare convergence, a float number
    """

    cnt = 0.0
    centroids = init_centroids(k,inputs)
    indices = assign_step(inputs, centroids)
    diff = 1
    while cnt < max_iter or diff > tol:
        cnt += 1.0
        oldmu = update_step(inputs, indices, k)
        indices = assign_step(inputs, oldmu)
        newmu = update_step(inputs, indices, k)
        diff = np.linalg.norm(newmu - oldmu)
        mu_converge = copy.deepcopy(newmu)
    return mu_converge        
            
            
