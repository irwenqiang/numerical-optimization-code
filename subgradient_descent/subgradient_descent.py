def subgradient_descent(function, subgradient, x0, alpha, iters=100):
    """Subgradient Descent

    Parameters
    ----------
    function : function
        Computes the objective function
    subgradient : function
        Computes a gradient for the objective function at x
    x0 : array
        initial value for x
    alpha : function
        function computing step sizes
    iters : int, optional
        number of iterations to perform

    Returns
    -------
    xs : list
        intermediate values for x
    """

    xs = [x0]
    x_best = x0

    for t in xrange(iters):
        x = xs[-1]
        g = subgradient(x)
        x_new = x - alpha(t, function(x), function(x_best), g) * g
        xs.append(x_new)
        if (function(x_new) < function(x_best)):
            x_best = x_new
        return xs

def polyak(t, f_x, f_x_best, g):
    if abs(g) > 0:
        return (f_x - f_x_best + 1.0 / (t+1)) / (g * g)
    else:
        return 0.0

if __name__ == "__main__":
    import os

    import numpy as np
    import pylab as pl
    import plotting as plotting

    ### SUBGRADIENT DESCENT ###
    function = np.abs
    subgradient = np.sign
    x0 = 0.75
    n_iterations = 200
    iterates = subgradient_descent(function, subgradient, x0, polyak, n_iterations)

    iterates = np.asarray(iterates)

    ### PLOTTING ###
    plotting.plot_iterates_vs_function(iterates, function,
                                     path='figures/iterates.png', y_star=0.0)
    plotting.plot_iteration_vs_function(iterates, function,
                                      path='figures/convergence.png', y_star=0.0)
