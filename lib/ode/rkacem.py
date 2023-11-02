import numpy as np
import math

def rkacem_system_method(T0, X0, h, getFunc, tol):
    """
        returns (Xh, h, h_next) where h is the time step used to advanced
        the iteration, Xh is value at T0+h, and h_next is the next approximated 
        optimum time step.
    """
    size = X0.shape[0]
    while True:
        K1 = np.zeros(size)
        for j in range(0, size):
            K1[j] = getFunc(j)(T0, X0)

        K2 = np.zeros(size)
        for j in range(0, size):
            K2[j] =  getFunc(j)(T0 + h/2, X0 + h*K1/2)

        K3 = np.zeros(size)
        for j in range(0, size):
            K3[j] = getFunc(j)(T0 + h/2, X0 + h*K2/2)

        K4 = np.zeros(size)
        for j in range(0, size):
            K4[j] = getFunc(j)(T0 + h, X0 + h*K3)
        XKh = X0 + h*(K1 + 2*K2+ 2*K3 + K4)/6

        SK1 = K1
        SK2 = K2

        SK3 = np.zeros(size)
        for j in range(0, size):
            SK3[j] = getFunc(j)(T0 + h/2, X0 + h*SK1/24 + (h*11*SK2)/24)

        SK4 = np.zeros(size)
        for j in range(0, size):
            SK4[j] = getFunc(j)(T0 + h, X0 + h*SK1/12 - (h*25*SK2)/132 + (h*73*SK3/66))

        XSh = X0 + ((2*h)/9) * ((SK1*SK1+SK1*SK2+SK2*SK2)/(SK1+SK2) + (SK2*SK2+SK2*SK3+SK3*SK3)/(SK2+SK3) + (SK3*SK3+SK3*SK4+SK4*SK4)/(SK3+SK4))

        errest = abs(XKh - XSh)* (281/13824)
        delta = .84 * ((tol/errest)**0.25)
        
        min_delta = np.min(delta)
        max_errest = np.max(errest)
        
        h_next = min_delta * h
        if max_errest < tol:
            break
        else:
            h = h_next
            
    return (XKh, h, h_next)