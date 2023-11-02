import numpy as np
import math

def rk55_system_method(T0, X0, h, getFunc, tol):
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
            K2[j] =  getFunc(j)(T0 + h/4, X0 + 1.5471214403*h*K1)
        K3 = np.zeros(size)
        for j in range(0, size):
            K3[j] =  getFunc(j)(T0 + h/2, X0 +  0.1756458393*h*K1 + 0.1243059001*h*K2)

        K4 = np.zeros(size)
        for j in range(0, size):
            K4[j] =  getFunc(j)(T0 + 3*h/4, X0 +  0.1009316694*h*K1 + 0.1100539630*h*K2 +  0.2890143692*h*K3)
            
        K5 = np.zeros(size)
        for j in range(0, size):
            K5[j] =  getFunc(j)(T0 + h, X0 +  0.99974318624*h*K1 - 0.0928890403*h*K2 - 0.6201812828*h*K3 + 0.7133271396*h*K4)
       
        XKh = X0 + h*( 0.2615038147*(K1 + K2)/2 - 0.2765809214*(K2 + K3)/2 +  0.5947141647*(K3 + K4)/2 +  0.4203629420*(K4 + K5)/2 )

        SK1 = K1
            
        SK2 = np.zeros(size)
        for j in range(0, size):
            SK2[j] =  getFunc(j)(T0 + h/4, X0 + 0.1017275411*h*SK1 )
            
        SK3 = np.zeros(size)
        for j in range(0, size):
            SK3[j] =  getFunc(j)(T0 + h/2, X0 - 0.5236574475*h*SK1 + 1.16533619101*h*SK2)
   


        SK4 = np.zeros(size)
        for j in range(0, size):
            SK4[j] = getFunc(j)(T0 + 3*h/4, X0 + 4.7450804540*h*SK1 - 4.2354437705*h*SK2 - 0.0096366835*h*SK3)
            
        SK5 = np.zeros(size)
        for j in range(0, size):
            SK5[j] = getFunc(j)(T0 + h, X0 - 0.5736403905*h*SK1 +   0.9301175162*h*SK2 + 0.4667978567*h*SK3 + 0.1767250176*h*SK4)

        XSh = X0 + (h)*( -0.1773157366*(SK1*SK1 + SK2*SK2)/(SK1+SK2) +  1.0254553152*(SK2*SK2 + SK3*SK3)/(SK2+SK3) - 0.0779114700*(SK3*SK3 + SK4*SK4)/(SK3+SK4) + 0.22977189140*(SK5*SK5 + SK4*SK4)/(SK5+SK4) )

        errest = abs(XKh - XSh)* (89/2042)
        delta = .84 * ((tol/errest)**0.25)
        
        min_delta = np.min(delta)
        max_errest = np.max(errest)
        
        h_next = min_delta * h
        if max_errest < tol:
            break
        else:
            h = h_next
            
    return (XKh, h, h_next)