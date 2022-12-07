import matplotlib.pyplot as plt
import numpy as np
from finite_difference import *


def error_plot(f, f_, x):
    x_input = np.geomspace(10**-1, 10**-10, 10)

    err1 = np.vectorize(lambda h: np.absolute(f_(x) - finite_difference_forward(f, x, h)))
    err2 = np.vectorize(lambda h: np.absolute(f_(x) - finite_difference_central(f, x, h)))

    plt.loglog(x_input, err1(x_input))
    plt.loglog(x_input, err2(x_input))
    plt.show()


if __name__ == "__main__":
    f = lambda x: x*np.exp(x)
    f_ = lambda x: (x+1)*np.exp(x)

    error_plot(f, f_, 2)
