from unittest import main, TestCase
from matrix_decomposition import *
import numpy as np

L = np.array([[1,0,0],                                                                                                                                                         
              [-2,1,0],                                                                                                                                                        
              [-2,-2,1]], dtype=np.double)
R = np.array([[2,-1,-2],                                                                                                                                                       
              [0,4,-1],                                                                                                                                                        
              [0,0,8]], dtype=np.double)
A = np.matmul(L,R)
b = np.array([1,1,1], dtype=np.double)

class TestMatrixDecomposition(TestCase):
    def test_forward_sub(self):
        global L, b
        myY = forward_sub(L, b)
        trueY = np.linalg.solve(L, b)
        np.testing.assert_array_equal(myY, trueY)

    def test_backward_sub(self):
        global L, R, b
        y = forward_sub(L, b)
        myX = backward_sub(R, y)
        trueX = np.linalg.solve(R, y)
        np.testing.assert_array_equal(myX, trueX)

    def test_lu_decomposition(self):
        global L, R, A
        newL, newR = lu_decomposition(A)
        np.testing.assert_array_equal(L, newL)
        np.testing.assert_array_equal(R, newR)

    def test_solve_with_lu(self):
        global A, b
        myX = solve_with_lu(A, b)
        trueX = np.linalg.solve(A, b)
        np.testing.assert_array_equal(myX, trueX)
if __name__ == "__main__":
    main()
