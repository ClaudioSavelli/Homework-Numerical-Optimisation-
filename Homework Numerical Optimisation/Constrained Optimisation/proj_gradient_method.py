import numpy as np
from numba import jit
from numba.typed import Dict
from function import *


@jit(nopython=True)
def project(x, X):
    for v in X.values():
        low = v[0]
        upp = v[1]
        start = int(v[2])
        stop = int(v[3])
        x[start:stop] = np.where(x[start:stop] > upp, upp, x[start:stop])
        x[start:stop] = np.where(x[start:stop] < low, low, x[start:stop])
    return x


#@jit(nopython=True)
def projected_gradient_bcktrck(x0, box, gamma, kmax, tolgrad, tolx, c1, rho, btmax, fin_diff, fd_type, h):
    
    ''' Function that performs the steepest descent optimization method for a given function.
    
    INPUTS:
    x0 = n−dimensional column vector;
    f = string that represent the function I want to use between the one stored there;
    alpha0 = the initial factor that multiplies the descent direction at each iteration;
    kmax = maximum number of iterations permitted;
    tolgrad = value used as stopping criterion w.r.t. the norm of the gradient;
    c1 = factor of the Armijo condition that must be a scalar in (0,1);
    rho = fixed factor, lesser than 1, used for reducing alpha0;
    btmax = maximum number of steps for updating alpha during the backtracking strategy.
    fin_diff = choose between using the finite differences method for the evaluation of the gradient or not
    fd_type = if fin_diff == True, choose between centered/forward/backword finite differences method
    
    OUTPUTS:
    xk = the last x computed by the function;
    fk = the value f(xk);
    gradfk_norm = value of the norm of gradf(xk)
    k = index of the last iteration performed
    x_seq = n−by−k matrix where the columns are the xk computed during the iterations
    bt_seq = k vector whose elements are the number of backtracking '''
    
    x_seq = np.empty((1, x0.shape[0]))
    x_seq[0] = x0.reshape(1, -1)
    bt_seq = np.empty((1, 1))
    f_seq = np.empty((1, 1))
    gradf_norm_seq = np.empty((1, 1))
    deltax_norm_seq = np.empty((1, 1))
    xk = x0
    fk = 0
    k = 0
    gradfk_norm = 0.0
    deltaxk_norm = 1.0
    alpha0 = 1.0
    alphak = alpha0
    
    fk = rosenbrock(xk)
    f_seq[0] = fk
    gradfk_norm = np.linalg.norm(grad_rosenbrock(xk, fin_diff, fd_type, h), 2)
    gradf_norm_seq[0] = gradfk_norm
    deltax_norm_seq[0] = deltaxk_norm
    
    while k < kmax and gradfk_norm > tolgrad and deltaxk_norm > tolx:
        gradfk = grad_rosenbrock(xk, fin_diff, fd_type, h)
        pk = -gradfk
        xhat = project(xk + gamma*pk, box)
        pik = xhat - xk
        xnew = xk + alphak*pik
        fnew = rosenbrock(xnew)
        
        bt = 0
        while bt < btmax and fnew > fk + c1*alphak*(gradfk @ pik):
            # update alpha
            alphak = rho*alphak
            xnew = xk + alphak*pik
            fnew = rosenbrock(xnew)
            bt = bt + 1
        alphak = alpha0
        
        deltaxk_norm = np.linalg.norm(xnew - xk, 2)
        deltax_norm_seq = np.append(deltax_norm_seq, np.array([[deltaxk_norm]]))
        xk = xnew
        fk = fnew
        gradfk_norm = np.linalg.norm(grad_rosenbrock(xk, fin_diff, fd_type, h), 2)
        x_seq = np.append(x_seq, xk.reshape(1, -1), axis=0)
        f_seq = np.append(f_seq, np.array([[fk]]))
        gradf_norm_seq = np.append(gradf_norm_seq, np.array([[gradfk_norm]]))
        print(k)
        if k == 0:
            bt_seq[0] = bt
        else:
            bt_seq = np.append(bt_seq, np.array([[bt]]))
        k = k + 1
        
    return x_seq, f_seq, gradf_norm_seq, deltax_norm_seq, k, bt_seq
