import numpy as np
from numba import jit
from numba.typed import Dict
from function import *


@jit(nopython=True)
def project(x, X):
    '''the general definition of the projection function.

    INPUTS: 
    x = points on which the projection is to be applied; 
    X = Constraints of the projection to be made and on the number of points;

    OUTPUTS: 
    x = points of input x projected;'''
    for v in X.values():
        low = v[0]
        upp = v[1]
        start = int(v[2])
        stop = int(v[3])
        x[start:stop] = np.where(x[start:stop] > upp, upp, x[start:stop])
        x[start:stop] = np.where(x[start:stop] < low, low, x[start:stop])
    return x


@jit(nopython=True)
def projected_gradient_bcktrck(x0, box, gamma, kmax, tolgrad, tolx, c1, rho, btmax, fin_diff, fd_type, h):
    
    ''' Function that performs the steepest descent optimization method for a given function.
    
    INPUTS:
    x0 = n−dimensional column vector;
    box =  the constraint box (in the report and elsewhere is the X)
    gamma = fixed factor gamma > 0 that multiplies the descent direction before the (possible) projection on delta(X), where X is the domain; 
    kmax = maximum number of iterations permitted;
    tolgrad = value used as stopping criterion w.r.t. the norm of the gradient;
    tolx = a real scalar value characterising the tol with respect to the norm of x_(k+1) - x_k to stop the method; 
    c1 = factor of the Armijo condition that must be a scalar in (0,1);
    rho = fixed factor, lesser than 1, used for reducing alpha0;
    btmax = maximum number of steps for updating alpha during the backtracking strategy;
    fin_diff = choose between using the finite differences method for the evaluation of the gradient or not;
    fd_type = if fin_diff == True, choose between centered/forward/backword finite differences method;
    h = the value of h previously evaluated to use in the evaluation of the gradient for finite difference method;
    
    OUTPUTS:
    x_seq = n−by−k matrix where the columns are the xk computed during the iterations; 
    f_seq = sequence of values of f(xk) computed during the iterations; 
    gradf_norm_seq = sequence of norm of grad f(xk) computed during the itarations; 
    k = number of iterations; 
    bt_seq = k vector whose elements are the number of backtracking '''
    
    x_seq = np.empty((kmax+1, x0.shape[0]))
    x_seq[0, :] = x0
    bt_seq = np.empty((kmax+1,))
    f_seq = np.empty((kmax+1,))
    gradf_norm_seq = np.empty((kmax+1,))
    deltax_norm_seq = np.empty((kmax+1,))
    xk = x0
    fk = 0
    k = 0
    gradfk_norm = 0.0
    deltaxk_norm = np.linalg.norm(x0, 2)
    alphak = gamma
    
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
        alphak = gamma
        
        deltaxk_norm = np.linalg.norm(xnew - xk, 2)
        deltax_norm_seq[k+1] = deltaxk_norm
        xk = xnew
        fk = fnew
        gradfk_norm = np.linalg.norm(grad_rosenbrock(xk, fin_diff, fd_type, h), 2)
        x_seq[k+1, :] = xk
        f_seq[k+1] = fk
        gradf_norm_seq[k+1] = gradfk_norm
        bt_seq[k] = bt
        # print(k)
        k = k + 1
        
    return x_seq, f_seq, gradf_norm_seq, deltax_norm_seq, k, bt_seq
