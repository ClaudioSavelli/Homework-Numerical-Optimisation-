import numpy as np
from functions import *


def steepest_descent_bcktrck(x0: np.ndarray, f: str, alpha0: float, kmax: int, tolgrad: float, c1: float, rho: float, btmax: int, fin_diff: bool):
    xseq = x0.reshape(-1,1)
    btseq = np.zeros((1,1))
    xk = x0.reshape(-1,1)
    fk = 0
    k = 0
    gradfk_norm = 0
    if f == 'Rosenbrock':
        fk = rosenbrock(xk)
        k = 0
        gradfk_norm = np.linalg.norm(grad_Rosenbrock(xk, fin_diff), 2)

        while k < kmax and gradfk_norm > tolgrad:
            pk = -grad_Rosenbrock(xk, fin_diff)
            bt = 0
            alphak = alpha0
            xnew = (xk + alphak*pk.reshape(-1,1)).reshape(-1,1)
            fnew = rosenbrock(xnew)
            gradfk = grad_Rosenbrock(xk, fin_diff)

            while (bt < btmax) and (fnew > fk + c1*alphak*np.dot(pk.flatten(), gradfk.flatten())):
                # update alpha
                alphak = rho*alphak
                xnew = (xk + alphak*pk.reshape(-1,1)).reshape(-1,1)
                fnew = rosenbrock(xnew)
                bt = bt + 1
            
            xk = xnew
            fk = fnew
            gradfk_norm = np.linalg.norm(grad_Rosenbrock(xk, fin_diff), 2)
            xseq = np.append(xseq, xk, axis=1)
            if btseq.size == 1:
                btseq[0,0] = bt
            else:
                btseq = np.append(btseq, np.array([[bt]]), axis=0)
            k = k + 1
    return xk, fk, gradfk_norm, k, xseq, btseq
