import numpy as np
import math

def rkcom_system_method(T0, X0, h, getFunc, tol):
    """
        returns (X1, K1) where X1 is value at T0+h, and K1 is slop at T0.
        K1 can be used to controll algorithm iteration
    """
    size = X0.shape[0]
    while True:
        # Todo perform the loop operation in more efficient way
        K1 = np.zeros(size)
        for j in range(0, size):
            K1[j] = getFunc(j)(T0, X0)

        K2 = np.zeros(size)
        for j in range(0, size):
            K2[j] =  getFunc(j)(T0 + h/2, X0 + h*K1/2)

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
            SK3[j] =  getFunc(j)(T0 + h/2, X0 + h*SK1/8 + (h*3*SK2)/8)

        SK4 = np.zeros(size)
        for j in range(0, size):
            SK4[j] = getFunc(j)(T0 + h, X0 + h*SK1/4 - (h*3*SK2)/4 + (h*3*SK3/2))

        XSh = X0 + (h/3)*(  (SK1*SK1 + SK2*SK2)/(SK1+SK2) +  (SK2*SK2 + SK3*SK3)/(SK2+SK3) +  (SK3*SK3 + SK4*SK4)/(SK3+SK4) )

        errest = abs(XKh - XSh)* (281/4608)
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