import numpy as np
from math import *
h = np.sqrt(np.finfo(float).eps/2)


def rosenbrock(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    #Evaluation of the Rosenbrock function in the point x
    return 100*(y - x**2)**2 + (1 - x)**2


def grad_rosenbrock(x: np.ndarray, y: np.ndarray, fin_diff: bool, type: str) -> np.ndarray:
    '''
    Compute the appoximation of the gradient via finite differences or with the true gradient
    
    INPUTS:
    x = ;
    y = ;
    fin_diff = choose between using the finite differences method for the evaluation of the gradient or not
    type = if fin_diff == True, choose between centered/forward/backword finite differences method

    OUTPUTS:
    gradfx=the appossimation of the gradient in x'''

    x_num = x.shape[1]
    y_num = y.shape[0]
    
    if (x.size == 1 and y.size == 1):
        grad = np.empty(2)
        if fin_diff == True:
            if (type == "fw" or type == "bw"): 
                fx = rosenbrock(x, y)[0, 0]
                if(type == "fw"): 
                    grad[0] = (rosenbrock(x+h, y) - fx)[0, 0] / h
                    grad[1] = (rosenbrock(x, y+h) - fx)[0, 0] / h
                else: 
                    grad[0] = -(rosenbrock(x-h, y) - fx)[0, 0] / h
                    grad[1] = -(rosenbrock(x, y-h) - fx)[0, 0] / h
            else:
                grad[0] = (rosenbrock(x+h, y) - rosenbrock(x-h, y))[0, 0] / (2*h)
                grad[1] = (rosenbrock(x, y+h) - rosenbrock(x, y-h))[0, 0] / (2*h)
        else:
            grad[0] = (400*x**3 - 400*x*y + 2*x - 2)[0, 0]
            grad[1] = (200*(y - x**2))[0, 0]
    else:
        grad = np.empty((2, y_num, x_num))
        if fin_diff == True:
            if (type == "fw" or type == "bw"): 
                fx = rosenbrock(x, y)
                if(type == "fw"): 
                    grad[0, :, :] = (rosenbrock(x+h, y) - fx) / h
                    grad[1, :, :] = (rosenbrock(x, y+h) - fx) / h
                else: 
                    grad[0, :, :] = -(rosenbrock(x-h, y) - fx) / h
                    grad[1, :, :] = -(rosenbrock(x, y-h) - fx) / h
            else:
                grad[0, :, :] = (rosenbrock(x+h, y) - rosenbrock(x-h, y)) / (2*h)
                grad[1, :, :] = (rosenbrock(x, y+h) - rosenbrock(x, y-h)) / (2*h)
        else:
            grad[0, :, :] = 400*x**3 - 400*x*y + 2*x - 2
            grad[1, :, :] = 200*(y - x**2)
    return grad


def extnd_powell(x: np.ndarray) -> float:
    #Evaluation of the Extended Powell function in the point x
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
    
    z = np.empty(num)
    for k in range(0, num):
        z[k] = f(x, k+1)
    return (0.5 * np.sum(z**2))


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

    grad = np.empty(num)
    if fin_diff == True:
        e = np.identity(num)
        if (type == "fw" or type == "bw"): 
            fx = extnd_powell(x)
        for i in range(0, num):
            if(type == "fw"): 
                grad[i] = (extnd_powell(x+h*e[i, :]) - fx) / h
            elif(type == "bw"): 
                grad[i] = -(extnd_powell(x-h*e[i, :]) - fx) / h
            else:
                grad[i] = (extnd_powell(x+h*e[i, :]) - extnd_powell(x-h*e[i, :])) / (2*h)
    else:
        for k in range(0, num):
            grad[k] = df(x, k+1)
    return grad


def banded_trig(x: np.ndarray) -> float:
    #Evaluation of the Banded trigonometric problem in the point x
    num = x.shape[0]
    if num < 2:
        raise Exception("Array length must be equal or higher than 2.")
    
    z = np.empty(num)
    
    #first iteration, different from the others 
    z[0] = 1 - cos(x[0]) - sin(x[1])
    
    for k in range(1, num-1):
        z[k] = (k + 1) * (1-cos(x[k]) + sin(x[k-1]) - sin(x[k+1]))
    
    #last iteration, different from the others
    z[num-1] = num * (1 - cos(x[num-1]) + sin(x[num-2]))
    return (np.sum(z))


def grad_banded_trig(x: np.ndarray, fin_diff: bool, type: str) -> np.ndarray:
    num = x.shape[0]
    if num < 2:
        raise Exception("Array length must be equal or higher than 2.")
    
    grad = np.empty(num)
    if fin_diff == True:
        e = np.identity(num)
        if (type == "fw" or type == "bw"): 
            fx = banded_trig(x)
        for i in range(0, num):
            if(type == "fw"): 
                grad[i] = (banded_trig(x+h*e[i, :]) - fx) / h
            elif(type == "bw"): 
                grad[i] = -(banded_trig(x-h*e[i, :]) - fx) / h
            else:
                grad[i] = (banded_trig(x+h*e[i, :]) - banded_trig(x-h*e[i, :])) / (2*h)
    else:
        #first iteration, different from the others 
        grad[0] = (sin(x[0]) + 2*cos(x[0]))
        
        for k in range(2, num):
            grad[k-1] = -(k-1)*cos(x[k-1]) + k*sin(x[k-1]) + (k+1)*cos(x[k-1])
        
        #last iteration, different from the others
        grad[num-1] = -(num-1)*cos(x[num-1]) + num*sin(x[num-1])
    return grad


def extnd_rosenb(x: np.ndarray) -> float:
    #Evaluation of the Extended Rosenbrock function in the point x
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
    
    z = np.empty(num)
    for k in range(0, num):
        z[k] = f(x, k+1)
    return (0.5 * np.sum(z**2))


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

    grad = np.empty(num)
    if fin_diff == True:
        e = np.identity(num)
        if (type == "fw" or type == "bw"): 
            fx = extnd_rosenb(x)
        for i in range(0, num):
            if(type == "fw"): 
                grad[i] = (extnd_rosenb(x+h*e[i, :]) - fx) / h
            elif(type == "bw"): 
                grad[i] = -(extnd_rosenb(x-h*e[i, :]) - fx) / h
            else:
                grad[i] = (extnd_rosenb(x+h*e[i, :]) - extnd_rosenb(x-h*e[i, :])) / (2*h)
    else:
        for k in range(0, num):
            grad[k] = df(x, k+1)
    return grad