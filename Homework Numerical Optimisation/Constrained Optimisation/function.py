import numpy as np
from numba import jit
from math import *


@jit(nopython=True)
def rosenbrock(x):
    #Evaluation of the Rosenbrock function in the point x
    n = x.shape[0] - 1
    def fk(x: np.ndarray, k: int) -> float:
        return 100 * (x[k+1] - x[k]**2)**2 + (x[k] - 1)**2
    z = np.empty(n)
    for k in range(0, n):
        z[k] = fk(x, k) 
    return np.sum(z)


@jit(nopython=True)
def grad_rosenbrock(x, fin_diff, type, h):
    '''
    Compute the appoximation of the gradient via finite differences or with the true gradient
    
    INPUTS:
    x = array of x-coordinates;
    fin_diff = choose between using the finite differences method for the evaluation of the gradient or not;
    type = if fin_diff == True, choose between centered/forward/backword finite differences method;
    h = the value of h previously evaluated to use in the evaluation of the gradient for finite difference method;

    OUTPUTS:
    gradfx=the appossimation of the gradient in x;
    '''
    num = x.shape[0]
    grad = np.empty(num)
    if fin_diff == True:
        if (type == "fw" or type == "bw"):
            fx = rosenbrock(x)
        for i in range(0, num):
            y = np.copy(x)
            if(type == "fw"): 
                y[i] += h
                grad[i] = (rosenbrock(y) - fx) / h
                y[i] -= h
            elif(type == "bw"):
                y[i] -= h 
                grad[i] = -(rosenbrock(y) - fx) / h
                y[i] += h
            else:
                y[i] += h
                z = np.copy(x)
                z[i] -= h
                grad[i] = (rosenbrock(y) - rosenbrock(z)) / (2*h)
                y[i] -= h
                z[i] += h 
    else:
        grad[0] = 400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] - 2
        for i in range(1, num-1):
            grad[i] = 400*x[i]**3 - 400*x[i]*x[i+1] - 200*x[i-1]**2 + 202*x[i] - 2
        grad[num-1] = 200*(x[num-1] - x[num-2]**2)
    return grad