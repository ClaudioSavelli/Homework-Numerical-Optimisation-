import numpy as np
from steepest_descent import *
import functions as funcs


def main():
    x0 = np.array([1, 2]).reshape(-1,1)
    alpha0 = 5
    tolgrad = 1e-12
    rho = 0.5
    c = 1e-4
    kmax = 35000
    btmax = 50
    fin_diff = True
    xk, fk, gradfk_norm, k, xseq, btseq = steepest_descent_bcktrck(x0, 'Rosenbrock', alpha0, kmax, tolgrad, c, rho, btmax, fin_diff)
    print(k, fk)
    print(xk)


if __name__ == '__main__':
    main()