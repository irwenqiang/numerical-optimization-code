def accelerated_gradient(gradient, x0, alpha, n_iterations=100):
    """Accelerated Gradient
    Parameters
    ----------
    gradient : function
        Computes the gradient of the objective function at x
    x0 : array
        initial value for x
    alpha : function
        Computes the step size
    n_iterations : int, optional
        number of iterations to perform

    Returns
    -------
    xs : list
        intermediate values for x
    """

    
    ys = [x0]
    xs = [x0]

    for t in xrange(n_iterations+1):
        y = ys[-1]
        x = xs[-1]
        g = gradient(y)
        x_new = y - alpha(y, g) * g
        y_new = x_new + ((t-1)/float(t+2)) * (x_new - x)
        ys.append(y_new)
        xs.append(x_new)
    return xs

class BacktrackingLineSearch(object):
    
    def __init__(self, function):
        self.function = function
        self.alpha = 0.05

    def __call__(self, y, g):
        f = self.function
        a = self.alpha
        while f(y - a * g) > f(y) - 0.5 * a * (g*g):
            a *= g
        return a

if __name__ == "__main__":
    import os

    import numpy as np
    import plotting as plotting

    ### ACCELERATED GRADIENT ###

    # problem definition
    function = lambda x: x ** 4     # the function to minimize
    gradient = lambda x: 4 * x **3  # its gradient
    alpha = BacktrackingLineSearch(function)
    x0 = 1.0
    n_iterations = 10

    # run accelerated gradient
    iterates = accelerated_gradient(gradient, x0, alpha, n_iterations=n_iterations)

    ### PLOTTING ###

    plotting.plot_iterates_vs_function(iterates, function,
                                     path='figures/iterates.png', y_star=0.0)

    plotting.plot_iteration_vs_function(iterates, function,
                                      path='figures/convergence.png', y_star=0.0)
    
