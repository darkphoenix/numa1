import numpy as np
from timeit import timeit
from scipy.linalg import null_space

def gen_google_matrix(alpha):
    L = np.array([[0,1,1,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,1,0,0,0,0,0,0,0],
                  [1,1,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,1,0,0,0,0,0,0,0],
                  [0,0,0,1,0,0,1,0,1,0,0,0],
                  [0,0,0,0,1,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,1,0,0,0,0,0,0,0],
                  [0,0,0,0,0,1,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,1,0,0,0,0],
                  [0,0,0,0,0,0,0,1,0,0,0,0],
                  [0,0,0,0,0,0,0,1,0,0,0,0]])
    L_ = np.zeros((12,12))
    for row in range(12):
        for col in range(12):
            if L.sum(axis=1)[row] == 0:
                L_[row, col] = 1
            else:
                L_[row, col] = L[row, col]
    
    #D = np.diag(L_.sum(axis=1))
    D_inv = np.diag(1/L_.sum(axis=1))
    #np.testing.assert_array_equal(np.linalg.inv(D), D_inv)

    A = (1-alpha)*np.matmul(np.transpose(L_),D_inv) + alpha/12*np.ones((12,12))

    return A

def l1(x):
    return np.sum(np.abs(x))

def power_method(A, v0, eps=1e-9):
    v = v0
    for i in range(100):
        v_last = v
        v = np.dot(A, v)


        if l1(v-v_last)/12 < eps:
            print("succeeded at %d"%i)
            print("v: ", v)
            print("order: ", np.argsort(v))
            return v
    
    print("stopped after 100")
    print(v)
    return v

def null_space_method(A):
        v = null_space(A)
        v = np.ndarray.flatten(v)
        v = v * np.sign(v)
        print("v: ", v)
        print("order: ", np.argsort(v))
        return v

if __name__=="__main__":
    for alpha in [0.1, 0.3, 0.6]:
        A = gen_google_matrix(alpha)
        v0 = 1/12*np.ones(12)
        print("attempting power method using alpha %f" % alpha)
        time = timeit(lambda: power_method(A, v0), number=1)
        print("took %fsec" %time)

        A_ = A-np.identity(12)

        print("attempting null space method using alpha %f" % alpha)
        time = timeit(lambda: null_space_method(A_), number=1)
        print("took %fsec" %time)