import numpy as np

def forward_sub(L, b):
    n = L.shape[0]
    y = np.zeros_like(b, dtype=np.double)

    y[0] = b[0] / L[0, 0]

    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i,:i], y[:i])) / L[i,i]

    return y

def backward_sub(U, y):
    n = U.shape[0]
    x = np.zeros_like(y, dtype=np.double)

    x[-1] = y[-1] / U[-1, -1]

    for i in range(n-2, -1, -1):
        x[i] = (y[i] - np.dot(U[i,i:], x[i:])) / U[i,i]

    return x

def lu_decomposition(A):
    n = A.shape[0]

    U = A.copy()
    L = np.eye(n, dtype=np.double)

    for i in range(n):
        factor = U[i+1:, i] / U[i, i]
        L[i+1:, i] = factor
        U[i+1:] -= factor[:, np.newaxis] * U[i]

    return L, U

def solve_with_lu(A, b):
    L, U = lu_decomposition(A)

    y = forward_sub(L, b)
    return backward_sub(U, y)
