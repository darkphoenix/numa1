import numpy as np

def kaczmarz_gs(A, b, x0, epsilon=0.1, kmax=1000):
    r = b - np.dot(A,x0)
    k = 0
    x = x0
    while np.linalg.norm(np.dot(np.transpose(A),r)) > epsilon and k < kmax:
        for j in range(x0.size):
            c = np.dot(np.transpose(A[:, j]), r)/(np.linalg.norm(A[:, j]))**2
            x[j] = x[j] + c
            r = r - np.dot(c, A[:, j])
        k += 1
    return k, x
