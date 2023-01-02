import numpy as np
from functions import *

def cgm_pol_rib(x0: np.ndarray, f: str, alpha0: float, kmax: int, tolgrad: float, c1: float, rho: float, btmax: int, fin_diff: bool, fd_type: str):
    
    ''' Function that performs the conjugate gradient method with Polak-Ribière for a given function.
    
    INPUTS:
    x0 = starting point;
    f = string that represent the function I want to use between the one stored there;
    alpha0 = the initial factor that multiplies the descent direction at each iteration;
    kmax = maximum number of iterations allowed;
    tolgrad = value used as stopping criterion considering the norm of the gradient;
    c1 = factor of the Armijo condition;
    rho = fixed factor used for reducing alpha0;
    btmax = maximum number of steps for updating alpha during the backtracking strategy.
    fin_diff = choose between using the finite differences method for the evaluation of the gradient or not
    fd_type = if fin_diff == True, choose between centered/forward/backword finite differences method
    
    OUTPUTS:
    x_seq = sequence of points xk computed during the iterations
    f_seq = sequence of values f(xk) evaluated during the iterations 
    grad_norm_seq = sequence values of the norms of gradf(xk) computed during the iterations 
    k = index of the last iteration performed
    bt_seq = sequence of the number of backtracking iterations done during the iterations '''
    
    #Initialisation of the parameters 
    x_seq = x0.reshape(1, -1)
    bt_seq = np.empty((1, 1))
    f_seq = np.empty((1, 1))
    gradf_norm_seq = np.empty((1, 1))
    xk = x0
    fk = 0
    betak = 0
    k = 0
    gradfk_norm = 0
    
    if f == 'Rosenbrock': #The function that we are going to evaluate is the Rosenbrock one 
        fk = rosenbrock(np.array([[xk[0]]]), np.array([[xk[1]]]))[0, 0] #evaluation of the function in xk
        f_seq[0] = fk #add fk in the sequence of f
        pk = -grad_rosenbrock(np.array([[xk[0]]]), np.array([[xk[1]]]), fin_diff, fd_type) #evaluate the gradient of the function (the direction of the first step)
        gradfk_norm = np.linalg.norm(grad_rosenbrock(np.array([[xk[0]]]), np.array([[xk[1]]]), fin_diff, fd_type), 2) #Evaluate the norm of the gradient of fk 
        #linarg is just a numpy library that contains linear algebra functions like the norm one
        gradf_norm_seq[0] = gradfk_norm #add the norm in the sequence of gradf_norm

        while k < kmax and gradfk_norm > tolgrad:
            bt = 0
            alphak = alpha0
            xnew = xk + alphak*pk #evaluate the new point following the direction pk
            fnew = rosenbrock(np.array([[xnew[0]]]), np.array([[xnew[1]]]))[0, 0] #evaluate the function in the new point 
            gradfk = grad_rosenbrock(np.array([[xk[0]]]), np.array([[xk[1]]]), fin_diff, fd_type) #evaluate the gradient in the new point

            while (bt < btmax) and (fnew > fk + c1*alphak*(gradfk @ pk)): #backtracking using the Armijo condition 
                # update alpha
                alphak = rho*alphak
                xnew = xk + alphak*pk
                fnew = rosenbrock(np.array([[xnew[0]]]), np.array([[xnew[1]]]))[0, 0]
                bt = bt + 1
            
            #The next point is found, now what we do is evaluating all the important informations for doing the analysis later and prepare the next iteration
            xk = xnew
            fk = fnew
            gradfnew = grad_rosenbrock(np.array([[xk[0]]]), np.array([[xk[1]]]), fin_diff, fd_type)
            betak = (gradfnew @ (gradfnew - gradfk)) / gradfk_norm**2
            pk = -gradfnew + betak*pk
            gradfk_norm = np.linalg.norm(grad_rosenbrock(np.array([[xk[0]]]), np.array([[xk[1]]]), fin_diff, fd_type), 2)
            x_seq = np.append(x_seq, xk.reshape(1, -1), axis=0)
            f_seq = np.append(f_seq, np.array([[fk]]))
            gradf_norm_seq = np.append(gradf_norm_seq, np.array([[gradfk_norm]]))
            if k == 0:
                bt_seq[0] = bt
            else:
                bt_seq = np.append(bt_seq, np.array([[bt]]))
            k = k + 1
            
    #The exact same steps were followed for the other three functions
    elif f == 'Extended Powell': #The function that we are going to evaluate is the Extended Powell one
        fk = extnd_powell(xk) 
        f_seq[0] = fk
        pk = -grad_extnd_powell(xk, fin_diff, fd_type) 
        gradfk_norm = np.linalg.norm(grad_extnd_powell(xk, fin_diff, fd_type), 2)
        gradf_norm_seq[0] = gradfk_norm

        while k < kmax and gradfk_norm > tolgrad:
            bt = 0
            alphak = alpha0
            xnew = xk + alphak*pk
            fnew = extnd_powell(xnew)
            gradfk = grad_extnd_powell(xk, fin_diff, fd_type)

            while (bt < btmax) and (fnew > fk + c1*alphak*(gradfk @ pk)):
                # update alpha
                alphak = rho*alphak
                xnew = xk + alphak*pk
                fnew = extnd_powell(xnew)
                bt = bt + 1
            
            xk = xnew
            fk = fnew
            gradfnew = grad_extnd_powell(xnew, fin_diff, fd_type)
            betak = (gradfnew @ (gradfnew - gradfk)) / gradfk_norm**2
            pk = -gradfnew + betak*pk
            gradfk_norm = np.linalg.norm(grad_extnd_powell(xk, fin_diff, fd_type), 2)
            x_seq = np.append(x_seq, xk.reshape(1, -1), axis=0)
            f_seq = np.append(f_seq, np.array([[fk]]))
            gradf_norm_seq = np.append(gradf_norm_seq, np.array([[gradfk_norm]]))
            if k == 0:
                bt_seq[0] = bt
            else:
                bt_seq = np.append(bt_seq, np.array([[bt]]))
            k = k + 1
            
    elif f == 'Extended Rosenbrock': #The function that we are going to evaluate is the Extended Rosenbrock one
        fk = extnd_rosenb(xk)
        f_seq[0] = fk
        pk = -grad_extnd_rosenb(xk, fin_diff, fd_type)
        gradfk_norm = np.linalg.norm(grad_extnd_rosenb(xk, fin_diff, fd_type), 2)
        gradf_norm_seq[0] = gradfk_norm

        while k < kmax and gradfk_norm > tolgrad:
            bt = 0
            alphak = alpha0
            xnew = xk + alphak*pk
            fnew = extnd_rosenb(xnew)
            gradfk = grad_extnd_rosenb(xk, fin_diff, fd_type)

            while (bt < btmax) and (fnew > fk + c1*alphak*(gradfk @ pk)):
                # update alpha
                alphak = rho*alphak
                xnew = xk + alphak*pk
                fnew = extnd_rosenb(xnew)
                bt = bt + 1
            
            xk = xnew
            fk = fnew
            gradfnew = grad_extnd_rosenb(xnew, fin_diff, fd_type)
            betak = (gradfnew @ (gradfnew - gradfk)) / gradfk_norm**2
            pk = -gradfnew + betak*pk
            gradfk_norm = np.linalg.norm(grad_extnd_rosenb(xk, fin_diff, fd_type), 2)
            x_seq = np.append(x_seq, xk.reshape(1, -1), axis=0)
            f_seq = np.append(f_seq, np.array([[fk]]))
            gradf_norm_seq = np.append(gradf_norm_seq, np.array([[gradfk_norm]]))
            if k == 0:
                bt_seq[0] = bt
            else:
                bt_seq = np.append(bt_seq, np.array([[bt]]))
            k = k + 1
    
    elif 'Banded Trigonometric': #The function that we are going to evaluate is the Banded Trigonometric one
        fk = banded_trig(xk)
        f_seq[0] = fk
        pk = -grad_banded_trig(xk, fin_diff, fd_type)
        gradfk_norm = np.linalg.norm(grad_banded_trig(xk, fin_diff, fd_type), 2)
        gradf_norm_seq[0] = gradfk_norm

        while k < kmax and gradfk_norm > tolgrad:
            bt = 0
            alphak = alpha0
            xnew = xk + alphak*pk
            fnew = banded_trig(xnew)
            gradfk = grad_banded_trig(xk, fin_diff, fd_type)

            while (bt < btmax) and (fnew > fk + c1*alphak*(gradfk @ pk)):
                # update alpha
                alphak = rho*alphak
                xnew = xk + alphak*pk
                fnew = banded_trig(xnew)
                bt = bt + 1
            
            xk = xnew
            fk = fnew
            gradfnew = grad_banded_trig(xnew, fin_diff, fd_type)
            betak = (gradfnew @ (gradfnew - gradfk)) / gradfk_norm**2
            pk = -gradfnew + betak*pk
            gradfk_norm = np.linalg.norm(grad_banded_trig(xk, fin_diff, fd_type), 2)
            x_seq = np.append(x_seq, xk.reshape(1, -1), axis=0)
            f_seq = np.append(f_seq, np.array([[fk]]))
            gradf_norm_seq = np.append(gradf_norm_seq, np.array([[gradfk_norm]]))
            if k == 0:
                bt_seq[0] = bt
            else:
                bt_seq = np.append(bt_seq, np.array([[bt]]))
            k = k + 1
            
    else:
        print(f"No function called {f} exists.")
        
    return x_seq, f_seq, gradf_norm_seq, k, bt_seq
