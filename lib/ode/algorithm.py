import numpy as np
import math

# ode_system_algorithm
def rk4_system_algorithm(t0, X0, tn, h, getFunc, ode_method, tol=100, max_tolerable_dynamic=-1):
    """
        stop algorithm iteration when max slop is less than stop_dynamic
        and stop at tn.
    """
    size = X0.shape[0]
    T = np.array([t0])
    XX = np.array([X0])
    
    i = 0
    max_abs_dynamic = math.inf
    while (T[i] < tn):
        i = i + 1

        Xh, h_prev, h = ode_method(T[i-1], XX[i-1], h, getFunc, tol)
        T = np.append(T, T[i-1]+h_prev)
        XX = np.append(XX, [Xh], axis=0)
        
        # return when slop is close to zero because if the slop is close to zero then 
        # system will not advance. so the current time value can be considered as final
        # time value.
        K = np.zeros(size)
        for j in range(0, size):
            K[j] = getFunc(j)(T[i], XX[i])
        max_dynamic = np.max(abs(K))
        if max_dynamic < max_tolerable_dynamic:
            return (T, XX)
        
    h = tn - T[i-1]
    Xh, h_prev, h = ode_method(T[i-1], XX[i-1], h, getFunc, tol)
    XX[i] = Xh
    T[i] = T[i-1] + h_prev

    return (T, XX)
