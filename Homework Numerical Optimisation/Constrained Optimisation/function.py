import numpy as np
from math import *


def rosenbrock(x: np.ndarray) -> float:
    #Evaluation of the Rosenbrock function in the point x
    n = x.shape[0] - 1
    def fk(x: np.ndarray, k: int) -> float:
        return 100 * (x[k+1] - x[k]**2)**2 + (x[k] - 1)**2
    z = np.empty(n)
    for k in range(0, n):
        z[k] = fk(x, k) 
    return np.sum(z)


def grad_rosenbrock(x: np.ndarray, fin_diff: bool, type: str, k: int) -> np.ndarray:
    '''
    Compute the appoximation of the gradient via finite differences or, when known, with the true gradient
    
    INPUTS:
    x = n−dimensional column vector;
    type = "centered" (Centered difference approximation for gradf), "forward" (Forward difference approximation for gradf), "backward" (Backward difference approximation for gradf);
    OUTPUTS:
    gradfx=the appossimation of the gradient in x via finite differences'''
    h = 10**(-k) * np.linalg.norm(x, 2)
    num = x.shape[0]
    grad = np.empty(num)
    if fin_diff == True:
        e = np.identity(num)
        if (type == "fw" or type == "bw"):
            fx = rosenbrock(x)
        for i in range(0, num):
            if(type == "fw"): 
                grad[i] = (rosenbrock(x + h*e[i, :]) - fx) / h
            elif(type == "bw"): 
                grad[i] = -(rosenbrock(x - h*e[i, :]) - fx) / h
            else:
                grad[i] = (rosenbrock(x + h*e[i, :]) - rosenbrock(x - h*e[i, :])) / (2*h)
    else:
        grad[0] = 400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] - 2
        for i in range(1, num-1):
            grad[i] = 400*x[i]**3 - 400*x[i]*x[i+1] - 200*x[i-1]**2 + 202*x[i] - 2
        grad[num-1] = 200*(x[num-1] - x[num-2]**2)
    return grad