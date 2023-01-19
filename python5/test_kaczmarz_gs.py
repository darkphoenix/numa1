from unittest import main, TestCase
from kaczmarz_gs import *
import numpy as np

class TestKaczmarzGS(TestCase):
    def test_kaczmarz_random(self):
        for N in range(4, 16):
            k = []
            for M in range(15, 27):
                A = np.random.random(size=(M,N))
                b = np.random.random(size=(M,))
                x0 = np.zeros(shape=(N,))

                numIter, myX = kaczmarz_gs(A, b, x0)
                print("%d x %d took %d iterations" %(M, N, numIter))
                k.append(numIter)

            print("k/N: %f" %(np.average(k)/N))

if __name__ == "__main__":
    main()
