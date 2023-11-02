import numpy as np
import math

def rkhm_system_method(T0, X0, h, getFunc, tol):
    """
        returns (Xh, h, h_next) where h is the time step used to advanced
        the iteration, Xh is value at T0+h, and h_next is the next approximated 
        optimum time step.
    """
    size = X0.shape[0]
    while True:
        K1 = np.zeros(size)
        for j in range(0, size):
            K1[j] = h * getFunc(j)(T0, X0)

        K2 = np.zeros(size)
        for j in range(0, size):
            K2[j] = h * getFunc(j)(T0 + h/2, X0 + K1/2)

        K3 = np.zeros(size)
        for j in range(0, size):
            K3[j] = h * getFunc(j)(T0 + h/2, X0 + K2/2)

        K4 = np.zeros(size)
        for j in range(0, size):
            K4[j] = h * getFunc(j)(T0 + h, X0 + K3)
        XKh = X0 + (K1 + 2*K2+ 2*K3 + K4)/6

        SK1 = K1
        SK2 = K2

        SK3 = np.zeros(size)
        for j in range(0, size):
            SK3[j] = h * getFunc(j)(T0 + h/2, X0 - SK1/8 + (5*SK2)/8)

        SK4 = np.zeros(size)
        for j in range(0, size):
            SK4[j] = h * getFunc(j)(T0 + h, X0 - SK1/4 - (7*SK2)/20 + (9*SK3/10))

        XSh = X0 + h*( SK2/6 + SK3/6 + (2*3)*(SK1*SK2)/(SK1+SK2) + (2*3)*(SK3*SK4)/(SK3+SK4) )

        errest = abs(XKh - XSh)* (5469/69120)
        delta = .84 * ((tol/errest)**0.25)
        
        min_delta = np.min(delta)
        max_errest = np.max(errest)
        
        h_next = min_delta * h
        if max_errest < tol:
            break
        else:
            h = h_next
            
    return (XKh, h, h_next)