import numpy as np
from math import *
h = np.sqrt(np.finfo(float).eps)


def rosenbrock(x: np.ndarray) -> float:
    #Evaluation of the Rosenbrock function in the point x
    return 100*(x[1, 0]-x[0, 0]**2)**2 + (1-x[0, 0])**2


def grad_rosenbrock(x: np.ndarray, fin_diff: bool, type: str) -> np.ndarray:
    '''
    Compute the appoximation of the gradient via finite differences or, when known, with the true gradient
    
    INPUTS:
    x = n−dimensional column vector;
    type = "centered" (Centered difference approximation for gradf), "forward" (Forward difference approximation for gradf), "backward" (Backward difference approximation for gradf);

    OUTPUTS:
    gradfx=the appossimation of the gradient in x via finite differences'''


    num = x.shape[0]
    grad = np.empty((num,1))
    if fin_diff == True:
        e = np.identity(num)
        if (type == "forward" or type == "backward"): 
            fx = rosenbrock(x)
        for i in range(0, num):
            if(type == "forward"): 
                grad[i, 0] = (rosenbrock(x+h*e[:, i].reshape(-1, 1)) - fx) / h
            elif(type == "backward"): 
                grad[i, 0] = -(rosenbrock(x-h*e[:, i].reshape(-1, 1)) - fx) / h
            else:
                grad[i, 0] = (rosenbrock(x+h*e[:, i].reshape(-1, 1)) - rosenbrock(x-h*e[:, i].reshape(-1, 1))) / (2*h)
    else:
        grad[0, 0] = 400*x[0, 0]**3 - 400*x[0, 0]*x[1, 0] + 2*x[0, 0] - 2
        grad[1, 0] = 200*(x[1, 0] - x[0, 0]**2)
    return grad


def extnd_powell(x: np.ndarray) -> float:
    num = x.shape[0]
    if num % 4 != 0:
        raise Exception("Array length must be multiple of 4.")
    
    def f(x: np.ndarray, k: int) -> float:
        match k % 4:
            case 1:
                k -= 1
                return x[k, 0] + 10*x[k+1, 0]
            case 2:
                k -= 1
                return sqrt(5)*(x[k+1, 0]-x[k+2, 0])
            case 3:
                k -= 1
                return (x[k-1, 0] - 2*x[k, 0])**2
            case 0:
                k -= 1
                return sqrt(10)*(x[k-3, 0] - x[k, 0])**2
    
    z = np.empty((num, 1))
    for k in range(0, num):
        z[k, 0] = f(x, k+1)
    return (0.5 * np.sum(z**2, axis=0))[0]


def grad_extnd_powell(x: np.ndarray, fin_diff: bool, type: str) -> np.ndarray:
    num = x.shape[0]
    if num % 4 != 0:
        raise Exception("Array length must be multiple of 4.")
    
    def df(x: np.ndarray, k: int) -> float:
        match k % 4:
            case 1:
                k -= 1
                return x[k, 0] + 10*x[k+1, 0] + 20*(x[k, 0] - x[k+3, 0])**3
            case 2:
                k -= 1
                return 10*(x[k-1, 0]+10*x[k, 0]) + 2*(x[k, 0]-2*x[k+1, 0])**3
            case 3:
                k -= 1
                return 5*(x[k, 0] - x[k+1, 0]) + 4*(2*x[k, 0] - x[k-1, 0])**3
            case 0:
                k -= 1
                return 5*(x[k, 0] - x[k-1, 0]) + 20*(x[k, 0] - x[k-3, 0])**3

    grad = np.empty((num, 1))
    if fin_diff == True:
        e = np.identity(num)
        if (type == "forward" or type == "backward"): 
            fx = extnd_powell(x)
        for i in range(0, num):
            if(type == "forward"): 
                grad[i, 0] = (extnd_powell(x+h*e[:, i].reshape(-1, 1)) - fx) / h
            elif(type == "backward"): 
                grad[i, 0] = -(extnd_powell(x-h*e[:, i].reshape(-1, 1)) - fx) / h
            else:
                grad[i, 0] = (extnd_powell(x+h*e[:, i].reshape(-1, 1)) - extnd_powell(x-h*e[:, i].reshape(-1, 1))) / (2*h)
    else:
        for k in range(0, num):
            grad[k, 0] = df(x, k+1)
    return grad

def banded_trig(x: np.ndarray) -> float:
    num = x.shape[0]
    z = np.empty((num, 1))
    
    #first iteration, different from the others 
    z[0] = (1-cos(x[0]-sin(x[1])))
    
    for k in range(1, num-1):
        z[k] = k*((1-cos(x[k])+sin(x[k-1]-sin(x[k+1]))))
    
    #last iteration, different from the others 
    n = num-1
    z[n] = (n)*((1-cos(x[n]))+sin(x[n-1]))
    return (np.sum(z, axis=0))[0]

def grad_banded_trig(x: np.ndarray, fin_diff: bool, type: str) -> np.ndarray:
    '''
    Compute the appoximation of the gradient via finite differences or, when known, with the true gradient
    
    INPUTS:
    x = n−dimensional column vector;
    type = "centered" (Centered difference approximation for gradf), "forward" (Forward difference approximation for gradf), "backward" (Backward difference approximation for gradf);

    OUTPUTS:
    gradfx=the appossimation of the gradient in x via finite differences'''


    num = x.shape[0]
    grad = np.empty((num,1))
    if fin_diff == True:
        e = np.identity(num)
        if (type == "forward" or type == "backward"): 
            fx = banded_trig(x)
        for i in range(0, num):
            if(type == "forward"): 
                grad[i] = (banded_trig(x+h*e[:, i].reshape(-1, 1)) - fx) / h
            elif(type == "backward"): 
                grad[i] = -(banded_trig(x-h*e[:, i].reshape(-1, 1)) - fx) / h
            else:
                grad[i] = (banded_trig(x+h*e[:, i].reshape(-1, 1)) - banded_trig(x-h*e[:, i].reshape(-1, 1))) / (2*h)
    else:
        #first iteration, different from the others 
        grad[0] = (sin(x[0]) + 2*cos(x[0]))
        
        for k in range(1, num-1):
            grad[k] = -(k-1)*cos(x[k]) + k*sin(x[k]) + (k+1)*cos(x[k])
        
        #last iteration, different from the others 
        k = num-1
        grad[k] = -(k-1)*(cos(x[k])) + k*(sin(x[k]))
    return grad