import numpy as np
import math
from .algorithm import rk4_system_algorithm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def test_ode_method(ode_method, step=.2, tol=.1):
    def getFunc(index):
        if (index == 0):
            return lambda t, X: X[0] - X[1] + 2
        if (index == 1):
            return lambda t, X: -X[0] + X[1] + 4*t

    T, XX = rk4_system_algorithm(0, np.array([-1, 0]), 1.5, step, getFunc, ode_method, tol=tol)    

    X = XX[:, 0]
    Y = XX[:, 1]

    plot_approximate_vs_exact((T, X), lambda x : -0.5*np.exp(2*x) + x**2 + 2*x - 0.5)
    plot_approximate_vs_exact((T, Y), lambda x : 0.5*np.exp(2*x) + x**2 - 0.5)


def plot_approximate_vs_exact(
    approximate,
    exact_fun,
    legend_loc="upper left",
    exact_plot_div=100
):
    X, Y = approximate
    plt.scatter(X, Y, label="approximate")

    EX_for_graph = np.linspace(X[0], X[-1], exact_plot_div)
    EY_for_graph = exact_fun(EX_for_graph)

    EY = exact_fun(X)
    rmse = math.sqrt(mean_squared_error(Y, EY))
    error = abs(Y[-1] - EY[-1])
    print("last point error =", error)
    print("rmse =", rmse)
    
    plt.plot(EX_for_graph, EY_for_graph, label="exact")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc=legend_loc)
    plt.show()