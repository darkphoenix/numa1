import numpy as np

if __name__ == '__main__':
    b = np.array([1,2,3])
    A = np.array([[1,2,3],
                  [4,5,6],
                  [7,8,9]])

    print(b, "= b")
    print(A, "= A")

    b1 = np.array([1,1,-1,-1])
    print(b1, "= b1")

    A1 = np.diag([5,5,5])
    A1 += np.diag([4,4],1)
    A1 += np.diag([3,3],-1)
    print(A1, "= A1")

    b2 = np.zeros(3, dtype=int)
    print(b2, "= b2")

    A2 = np.ones((4,2), dtype=int)
    print(A2, "= A2")

    print(b1[:0], "= b1[:1]")
    print(b2[-1:], "= b2[-1:]")

    print(A1[0, :], "= A1[1, :]")
    print(A2[:, -1], "=A2[:, -1]")
