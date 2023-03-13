import numpy as np
import math

def rkacem_system_method(T0, X0, h, getFunc, tol):
    """
        returns (X1, K1) where X1 is value at T0+h, and K1 is slop at T0.
        K1 can be used to controll algorithm iteration
    """
    size = X0.shape[0]
    while True:
        # Todo perform the loop operation in more efficient way
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
            SK3[j] = h * getFunc(j)(T0 + h/2, X0 + SK1/24 + (11*SK2)/24)

        SK4 = np.zeros(size)
        for j in range(0, size):
            SK4[j] = h * getFunc(j)(T0 + h/2, X0 + SK1/12 - (25*SK2)/132 + (73*SK3/66))

        XSh = X0 + ((2*h)/9) * ((SK1*SK1+SK1*SK2+SK2*SK2)/(SK1+SK2) + (SK2*SK2+SK2*SK3+SK3*SK3)/(SK2+SK3) + (SK3*SK3+SK3*SK4+SK4*SK4)/(SK3+SK4))

        errest = abs(XKh - XSh)* (281/13824)
        # print(errest)
        delta = .84 * ((tol/errest)**0.25)
        
        #Todo try to make a algorithm without the below line
        min_delta = np.min(delta)
        max_errest = np.max(errest)
        
        h_next = min_delta * h
        if max_errest < tol:
            break
        else:
            h = h_next
            
    return (XKh, h, h_next)