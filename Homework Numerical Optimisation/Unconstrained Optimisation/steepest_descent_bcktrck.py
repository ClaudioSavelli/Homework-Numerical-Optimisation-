import numpy as np
from functions import *

def steepest_descent_bcktrck(x0: np.ndarray, f: str, alpha0: float, kmax: int, tolgrad: float, c1: float, rho: float, btmax: int, fin_diff: bool, fd_type: str):
    
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
    
    x_seq = x0.reshape(1, -1)
    bt_seq = np.empty((1, 1))
    f_seq = np.empty((1, 1))
    gradf_norm_seq = np.empty((1, 1))
    xk = x0
    fk = 0
    k = 0
    alphak = alpha0
    gradfk_norm = 0
    
    if f == 'Rosenbrock':
        fk = rosenbrock(xk)
        k = 0
        gradfk_norm = np.linalg.norm(grad_rosenbrock(xk, fin_diff, fd_type), 2)

        while k < kmax and gradfk_norm > tolgrad:
            gradfk = grad_rosenbrock(xk, fin_diff, fd_type)
            pk = -gradfk
            xnew = xk + alphak*pk
            fnew = rosenbrock(xnew)
            bt = 0
            alphak = alpha0

            while (bt < btmax) and (fnew > fk + c1*alphak*(gradfk @ pk)):
                # update alpha
                alphak = rho*alphak
                xnew = xk + alphak*pk
                fnew = rosenbrock(xnew)
                bt = bt + 1
            
            xk = xnew
            fk = fnew
            gradfk_norm = np.linalg.norm(grad_rosenbrock(xk, fin_diff, fd_type), 2)
            x_seq = np.append(x_seq, xk.reshape(1, -1), axis=0)
            if k == 0:
                bt_seq[0] = bt
                f_seq[0] = fk
                gradf_norm_seq[0] = gradfk_norm
            else:
                bt_seq = np.append(bt_seq, np.array([[bt]]))
                f_seq = np.append(f_seq, np.array([[fk]]))
                gradf_norm_seq = np.append(gradf_norm_seq, np.array([[gradfk_norm]]))
            k = k + 1
            
    elif f == 'Extended Powell':
        fk = extnd_powell(xk)
        k = 0
        gradfk_norm = np.linalg.norm(grad_extnd_powell(xk, fin_diff, fd_type), 2)

        while k < kmax and gradfk_norm > tolgrad:
            gradfk = grad_extnd_powell(xk, fin_diff, fd_type)
            pk = -gradfk
            xnew = xk + alphak*pk
            fnew = extnd_powell(xnew)
            bt = 0
            alphak = alpha0

            while (bt < btmax) and (fnew > fk + c1*alphak*(gradfk @ pk)):
                # update alpha
                alphak = rho*alphak
                xnew = xk + alphak*pk
                fnew = extnd_powell(xnew)
                bt = bt + 1
            
            xk = xnew
            fk = fnew
            gradfk_norm = np.linalg.norm(grad_extnd_powell(xk, fin_diff, fd_type), 2)
            x_seq = np.append(x_seq, xk.reshape(1, -1), axis=0)
            if k == 0:
                bt_seq[0] = bt
                f_seq[0] = fk
                gradf_norm_seq[0] = gradfk_norm
            else:
                bt_seq = np.append(bt_seq, np.array([[bt]]))
                f_seq = np.append(f_seq, np.array([[fk]]))
                gradf_norm_seq = np.append(gradf_norm_seq, np.array([[gradfk_norm]]))
            #print(k)
            k = k + 1
            
    elif f == 'Extended Rosenbrock':
        fk = extnd_rosenb(xk)
        k = 0
        gradfk_norm = np.linalg.norm(grad_extnd_rosenb(xk, fin_diff, fd_type), 2)

        while k < kmax and gradfk_norm > tolgrad:
            gradfk = grad_extnd_rosenb(xk, fin_diff, fd_type)
            pk = -gradfk
            xnew = xk + alphak*pk
            fnew = extnd_rosenb(xnew)
            bt = 0
            alphak = alpha0

            while (bt < btmax) and (fnew > fk + c1*alphak*(gradfk @ pk)):
                # update alpha
                alphak = rho*alphak
                xnew = xk + alphak*pk
                fnew = extnd_rosenb(xnew)
                bt = bt + 1
            
            xk = xnew
            fk = fnew
            gradfk_norm = np.linalg.norm(grad_extnd_rosenb(xk, fin_diff, fd_type), 2)
            x_seq = np.append(x_seq, xk.reshape(1, -1), axis=0)
            if k == 0:
                bt_seq[0] = bt
                f_seq[0] = fk
                gradf_norm_seq[0] = gradfk_norm
            else:
                bt_seq = np.append(bt_seq, np.array([[bt]]))
                f_seq = np.append(f_seq, np.array([[fk]]))
                gradf_norm_seq = np.append(gradf_norm_seq, np.array([[gradfk_norm]]))
            k = k + 1
    
    elif 'Banded Trigonometric':
        fk = banded_trig(xk)
        k = 0
        gradfk_norm = np.linalg.norm(grad_banded_trig(xk, fin_diff, fd_type), 2)

        while k < kmax and gradfk_norm > tolgrad:
            gradfk = grad_banded_trig(xk, fin_diff, fd_type)
            pk = -gradfk
            xnew = xk + alphak*pk
            fnew = banded_trig(xnew)
            bt = 0
            alphak = alpha0

            while (bt < btmax) and (fnew > fk + c1*alphak*(gradfk @ pk)):
                # update alpha
                alphak = rho*alphak
                xnew = xk + alphak*pk
                fnew = banded_trig(xnew)
                bt = bt + 1
            
            xk = xnew
            fk = fnew
            gradfk_norm = np.linalg.norm(grad_banded_trig(xk, fin_diff, fd_type), 2)
            x_seq = np.append(x_seq, xk.reshape(1, -1), axis=0)
            if k == 0:
                bt_seq[0] = bt
                f_seq[0] = fk
                gradf_norm_seq[0] = gradfk_norm
            else:
                bt_seq = np.append(bt_seq, np.array([[bt]]))
                f_seq = np.append(f_seq, np.array([[fk]]))
                gradf_norm_seq = np.append(gradf_norm_seq, np.array([[gradfk_norm]]))
            k = k + 1
            
    else:
        print(f"No function called {f} exists.")
        
    return xk, f_seq, gradf_norm_seq, k, x_seq, bt_seq
