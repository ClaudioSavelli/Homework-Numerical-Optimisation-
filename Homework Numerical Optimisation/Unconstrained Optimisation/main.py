import numpy as np
from steepest_descent_bcktrck import *
import functions as funcs
from sklearn.model_selection import ParameterGrid


def main():
    #Let's define all the constants necessary for the evaluation of the methods we are going to use
    x0 = np.array([1.2, 1.2]).reshape(-1,1)
    x1 = np.array([-1.2, 1]).reshape(-1,1)
    alpha0 = 5
    tolgrad = 1e-12
    rho = 0.5
    c = 1e-4
    kmax = 1000
    btmax = 50
    fin_diff = True
    
    #Let's test the Steepest descent method with Backtrack using the Armijo condition to see what is the result obtained 
    xk, fk, gradfk_norm, k, xseq, btseq = steepest_descent_bcktrck(x0, 'Rosenbrock', alpha0, kmax, tolgrad, c, rho, btmax, fin_diff)
    print("Analysis of the point x0 = ", x0.reshape(1,-1))
    print("Number of iterations performed: ", k)
    print("Evaluation of Rosembrook function in the reached point: ", fk)
    print("Actual coordinates of the reached point xk: ", xk.reshape(1,-1))
    print("\n")
    
    xk, fk, gradfk_norm, k, xseq, btseq = steepest_descent_bcktrck(x1, 'Rosenbrock', alpha0, kmax, tolgrad, c, rho, btmax, fin_diff)
    print("Analysis of the point x0 = ", x1.reshape(1,-1))
    print("Number of iterations performed: ", k)
    print("Evaluation of Rosembrook function in the reached point: ", fk)
    print("Actual coordinates of the reached point xk: ", xk.reshape(1,-1))
    print("\n")
    
    #Tuning of parameters to see the different results 

    params = {"c": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
              "rho": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}
    beskfK = 1
    
    for param in ParameterGrid(params): 
        xk, fk, gradfk_norm, k, xseq, btseq = steepest_descent_bcktrck(x0, 'Rosenbrock', alpha0, kmax, tolgrad, param["c"], param["rho"], btmax, fin_diff)
        if (fk < beskfK): 
            bestXk = xk
            bestK = k
            beskfK = fk 
            bestParam = param
        
    print("The best parameters for this evaluation are: ", bestParam)    
    print("Analysis of the point x0 = ", x0.reshape(1,-1))
    print("Number of iterations performed: ", bestK)
    print("Evaluation of Rosembrook function in the reached point: ", beskfK)
    print("Actual coordinates of the reached point xk: ", bestXk.reshape(1,-1))
    print("\n")
    #The best c is always the higher one (0.1), for rho instead it depends, because for the point x0 the best one is 0.3, for x1 instead 0.5


if __name__ == '__main__':
    main()