import numpy as np
import math

def rk4_system_method(T0, X0, h, getFunc, tol=100):
    """
        returns (X1, K1) where X1 is value at T0+h, and K1 is slop at T0.
        K1 can be used to controll algorithm iteration
    """
    system_length = X0.shape[0]
    K1 = np.zeros(system_length)
    
    # Todo perform the loop operation in more efficient way
    for j in range(0, system_length):
        K1[j] = h * getFunc(j)(T0, X0)

    K2 = np.zeros(system_length)
    for j in range(0, system_length):
        K2[j] = h * getFunc(j)(T0 + h/2, X0 + K1/2)

    K3 = np.zeros(system_length)
    for j in range(0, system_length):
        K3[j] = h * getFunc(j)(T0 + h/2, X0 + K2/2)

    K4 = np.zeros(system_length)
    for j in range(0, system_length):
        K4[j] = h * getFunc(j)(T0 + h, X0 + K3)
    Xh = X0 + (K1 + 2*K2+ 2*K3 + K4)/6
    return (Xh, h, h)

