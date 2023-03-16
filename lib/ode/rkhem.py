import numpy as np
import math
from numpy import sqrt
from numpy import abs
def rkhem_system_method(T0, X0, h, getFunc, tol):
    """
        returns (X1, K1) where X1 is value at T0+h, and K1 is slop at T0.
        K1 can be used to controll algorithm iteration
    """

    size = X0.shape[0]
    while True:
        # Todo perform the loop operation in more efficient way
        
        K1 = np.zeros(size)
        for j in range(0, size):
            K1[j] =  getFunc(j)(T0, X0)

        K2 = np.zeros(size)
        for j in range(0, size):
            K2[j] = getFunc(j)(T0 + h/2, X0 + h*K1/2)

        K3 = np.zeros(size)
        for j in range(0, size):
            K3[j] =  getFunc(j)(T0 + h/2, X0 + h*K2/2)

        K4 = np.zeros(size)
        for j in range(0, size):
            K4[j] =  getFunc(j)(T0 + h, X0 + h*K3)
        XKh = X0 + h*(K1 + 2*K2+ 2*K3 + K4)/6

        SK1 = K1
        SK2 = K2

        SK3 = np.zeros(size)
        for j in range(0, size):
            SK3[j] =  getFunc(j)(T0 + h/2, X0 - h*SK1/44 + (h*25*SK2)/48)

        SK4 = np.zeros(size)
        for j in range(0, size):
            SK4[j] = getFunc(j)(T0 + h, X0 - h*SK1/24 + (h*47*SK2)/600 + (h*289*SK3/300))

        XSh = X0 + (h/9) * (SK1 + 2*(SK2 + SK3) + SK4 + sqrt(abs(SK1*SK2)) + sqrt(abs(SK2*SK3)) + sqrt(abs(SK3*SK4)))

        errest = abs(XKh - XSh)* (121809/1658880)
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