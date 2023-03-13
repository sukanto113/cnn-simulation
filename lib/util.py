import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def plot_XY_function_graph(x0, xn, f):
    x = np.linspace(x0, xn, 100)
    y = f(x)
    plt.plot(x, y)
    plt.show()