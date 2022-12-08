import numpy as np
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
    grad = np.zeros((num,1))
    e = np.identity(num)
    if fin_diff == True:
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