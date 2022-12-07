import numpy as np
def finite_difference_forward(f, x, h):
    return (f(x+h)-f(x))/h

def finite_difference_central(f, x, h):
    return (f(x+h)-f(x-h))/(2*h)
