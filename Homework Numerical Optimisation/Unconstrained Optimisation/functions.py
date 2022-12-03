import numpy as np
h = np.sqrt(np.finfo(float).eps)

def rosenbrock(x: np.ndarray) -> float:
    return 100*(x[1,0]-x[0,0]**2)**2 + (1-x[0,0])**2

def grad_Rosenbrock(x: np.ndarray, fin_diff: bool) -> np.ndarray:
    num = x.shape[0]
    grad = np.zeros((num,1))
    e = np.identity(num)
    if fin_diff == True:
        #fx = Rosenbrock(x)
        for i in range(0, num):
            grad[i,0] = (rosenbrock(x+h*e[:,i].reshape(-1,1)) - rosenbrock(x-h*e[:,i].reshape(-1,1))) / (2*h)
            # grad[i] = (Rosenbrock(x+h*e[:,i]) - fx) / h
            # grad[i] = -(Rosenbrock(x-h*e[:,i]) - fx) / h
    else:
        grad[0,0] = 400*x[0,0]**3 - 400*x[0,0]*x[1,0] + 2*x[0,0] - 2
        grad[1,0] = 200*(x[1,0] - x[0,0]**2)
    return grad