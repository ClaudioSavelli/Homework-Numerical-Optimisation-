import numpy as np
from math import *
h = np.sqrt(np.finfo(float).eps)


def rosenbrock(x: np.ndarray) -> float:
    #Evaluation of the Rosenbrock function in the point x
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2


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
        if (type == "fw" or type == "bw"): 
            fx = rosenbrock(x)
        for i in range(0, num):
            if(type == "fw"): 
                grad[i] = (rosenbrock(x+h*e[:, i].reshape(-1, 1)) - fx) / h
            elif(type == "bw"): 
                grad[i] = -(rosenbrock(x-h*e[:, i].reshape(-1, 1)) - fx) / h
            else:
                grad[i] = (rosenbrock(x+h*e[:, i].reshape(-1, 1)) - rosenbrock(x-h*e[:, i].reshape(-1, 1))) / (2*h)
    else:
        grad[0] = 400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] - 2
        grad[1] = 200*(x[1] - x[0]**2)
    return grad


def extnd_powell(x: np.ndarray) -> float:
    num = x.shape[0]
    if num % 4 != 0:
        raise Exception("Array length must be multiple of 4.")
    
    def f(x: np.ndarray, k: int) -> float:
        match k % 4:
            case 1:
                k -= 1
                return x[k] + 10*x[k+1]
            case 2:
                k -= 1
                return sqrt(5)*(x[k+1]-x[k+2])
            case 3:
                k -= 1
                return (x[k-1] - 2*x[k])**2
            case 0:
                k -= 1
                return sqrt(10)*(x[k-3] - x[k])**2
    
    z = np.empty((num, 1))
    for k in range(0, num):
        z[k] = f(x, k+1)
    return (0.5 * np.sum(z**2, axis=0))[0]


def grad_extnd_powell(x: np.ndarray, fin_diff: bool, type: str) -> np.ndarray:
    num = x.shape[0]
    if num % 4 != 0:
        raise Exception("Array length must be multiple of 4.")
    
    def df(x: np.ndarray, k: int) -> float:
        xk = x[k-1]
        match k % 4:
            case 1:
                k -= 1
                return xk + 10*x[k+1] + 20*(xk - x[k+3])**3
            case 2:
                k -= 1
                return 10*(x[k-1]+10*xk) + 2*(xk-2*x[k+1])**3
            case 3:
                k -= 1
                return 5*(xk - x[k+1]) + 4*(2*xk - x[k-1])**3
            case 0:
                k -= 1
                return 5*(xk - x[k-1]) + 20*(xk - x[k-3])**3

    grad = np.empty((num, 1))
    if fin_diff == True:
        e = np.identity(num)
        if (type == "fw" or type == "bw"): 
            fx = extnd_powell(x)
        for i in range(0, num):
            if(type == "fw"): 
                grad[i] = (extnd_powell(x+h*e[:, i].reshape(-1, 1)) - fx) / h
            elif(type == "bw"): 
                grad[i] = -(extnd_powell(x-h*e[:, i].reshape(-1, 1)) - fx) / h
            else:
                grad[i] = (extnd_powell(x+h*e[:, i].reshape(-1, 1)) - extnd_powell(x-h*e[:, i].reshape(-1, 1))) / (2*h)
    else:
        for k in range(0, num):
            grad[k] = df(x, k+1)
    return grad


def banded_trig(x: np.ndarray) -> float:
    num = x.shape[0]
    if num < 2:
        raise Exception("Array length must be equal or higher than 2.")
    
    z = np.empty((num, 1))
    
    #first iteration, different from the others 
    z[0] = 1 - cos(x[0]) - sin(x[1])
    
    for k in range(1, num-1):
        z[k] = (k + 1) * (1-cos(x[k]) + sin(x[k-1]) - sin(x[k+1]))
    
    #last iteration, different from the others
    z[num-1] = num * (1 - cos(x[num-1]) + sin(x[num-2]))
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
    if num < 2:
        raise Exception("Array length must be equal or higher than 2.")
    
    grad = np.empty((num,1))
    if fin_diff == True:
        e = np.identity(num)
        if (type == "fw" or type == "bw"): 
            fx = banded_trig(x)
        for i in range(0, num):
            if(type == "fw"): 
                grad[i] = (banded_trig(x+h*e[:, i].reshape(-1, 1)) - fx) / h
            elif(type == "bw"): 
                grad[i] = -(banded_trig(x-h*e[:, i].reshape(-1, 1)) - fx) / h
            else:
                grad[i] = (banded_trig(x+h*e[:, i].reshape(-1, 1)) - banded_trig(x-h*e[:, i].reshape(-1, 1))) / (2*h)
    else:
        #first iteration, different from the others 
        grad[0] = (sin(x[0]) + 2*cos(x[0]))
        
        for k in range(2, num):
            grad[k-1] = -(k-1)*cos(x[k-1]) + k*sin(x[k-1]) + (k+1)*cos(x[k-1])
        
        #last iteration, different from the others
        grad[num-1] = -(num-1)*cos(x[num-1]) + num*sin(x[num-1])
    return grad


def extnd_rosenb(x: np.ndarray) -> float:
    num = x.shape[0]
    if num % 2 != 0:
        raise Exception("Array length must be multiple of 2.")
    
    def f(x: np.ndarray, k: int) -> float:
        match k % 2:
            case 1:
                k -= 1
                return 10*(x[k]**2 - x[k+1])
            case 0:
                k -= 1
                return x[k-1] - 1
    
    z = np.empty((num, 1))
    for k in range(0, num):
        z[k] = f(x, k+1)
    return (0.5 * np.sum(z**2, axis=0))[0]


def grad_extnd_rosenb(x: np.ndarray, fin_diff: bool, type: str) -> np.ndarray:
    num = x.shape[0]
    if num % 2 != 0:
        raise Exception("Array length must be multiple of 2.")
    
    def df(x: np.ndarray, k: int) -> float:
        xk = x[k-1]
        match k % 2:
            case 1:
                k -= 1
                return 200*xk**3 - 200*xk*x[k+1] + xk - 1
            case 0:
                k -= 1
                return 100*(xk - x[k-1]**2)

    grad = np.empty((num, 1))
    if fin_diff == True:
        e = np.identity(num)
        if (type == "fw" or type == "bw"): 
            fx = extnd_rosenb(x)
        for i in range(0, num):
            if(type == "fw"): 
                grad[i] = (extnd_rosenb(x+h*e[:, i].reshape(-1, 1)) - fx) / h
            elif(type == "bw"): 
                grad[i] = -(extnd_rosenb(x-h*e[:, i].reshape(-1, 1)) - fx) / h
            else:
                grad[i] = (extnd_rosenb(x+h*e[:, i].reshape(-1, 1)) - extnd_rosenb(x-h*e[:, i].reshape(-1, 1))) / (2*h)
    else:
        for k in range(0, num):
            grad[k] = df(x, k+1)
    return grad