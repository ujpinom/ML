from scipy.spatial.distance import pdist,squareform
from scipy.linalg import eigh
import numpy as np

def rbf_kernel(X,gamma, n_components):

    sq_dis = pdist(X, 'sqeuclidean')

    matriz_sq_distance = squareform(sq_dis)
    K = np.exp(-gamma*matriz_sq_distance)
    N=K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    X_pc = np.column_stack((eigvecs[:, i]
                            for i in range(n_components)))
    return X_pc

