import numpy as np
from numpy.random import random_sample, laplace
from scipy.spatial.distance import cdist
import sys, getopt
import matplotlib.pyplot as plt


'''
    differentially private k-means clustering
    input
        X: N x 2 matrix, each row corresponds to one datapoint which lies within [-1, 1] x [-1, 1]
        k: number of clusters
        t: number of iterations
        e: epsilon (if e is none, no DP mechanism is applied)
    output
        Y: N-dimensional vector, each dimension is the cluster assignment of the corresponding datapoint
'''
def laplace_mech_vec(qs, sensitivity, epsilon):
    return qs + np.random.laplace(loc=0, scale=sensitivity / epsilon, size=len(qs))

def laplace_mech(v, sensitivity, epsilon):
    return v + np.random.laplace(loc=0, scale=sensitivity / epsilon)

def dp_kmeans(X, k, t, e):

    # randomly initialize the centroids within [-1, 1] x [-1, 1]
    centroids = random_sample((k, 2))*2 - 1
    
    for i in range(t):

        # compute the pairwise distance between each datapoint and each centroid
        pair_dist = cdist(X, centroids)
        
        # assign datapoints to their nearest centroids
        Y = np.argmin(pair_dist, axis=1)
        
        if i == t-1:
            return Y
        
        # compute new centroids 
        if e == None:
            # baseline version
            for j in range(k):
                centroids[j] = np.mean(X[Y == j], axis=0)
        else:
            # differentially private version
            for j in range(k):
                
                #!!!todo
                # compute the noisy size of the j-th cluster
                # hint: np.sum(Y == j) + noise, 
                #       where noise is sampled from Laplace distribution: laplace(scale = lambda)
                
                #!!!todo
                # compute the noisy sum of the j-th cluster along each dimension 
                # hint: np.sum(X[Y == j][0]) + noise and np.sum(X[Y == j][1]) + noise (or combine them together, even better :)
                #       where noise is sampled from Laplace distribution: laplace(scale = lambda)
                
                #!!!todo
                # update the j-th centroid 
                # hint: noisy sum divided by noisy size
                
        
if __name__ == '__main__':
    
    d = 'data.npy'
    k = 5
    t = 10
    e = 1

    # parse arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:k:t:e:")
    except getopt.GetoptError:
        print('dp_kmeans.py -d <data_file> -k <cluster_num> -t <iteration_num> -e <epsilon> (optional)')
        sys.exit(2)
    for opt, arg in opts:
        print(opt, arg)
        if opt == '-d':
            d = arg
        elif opt == '-k':
            k = int(arg)
        elif opt == '-t':
            t = int(arg)
        elif opt == '-e':
            e = float(arg)
        
    # load the data file 
    X = np.load(d)
    
    # run k-means clustering
    Y = dp_kmeans(X, k, t, e)
    
    # visualize the results
    plt.plot()
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.title('dp-kmeans (e = {})'.format(e))
    plt.show()


