def gradient_descent(gradient, x0, alpha, iters=100):
    """Gradient descent
    Parameters
    ----------
    gradient : function
        Computes the gradient of the objective function at x
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

    for t in xrange(iters):
        x = xs[-1]
        g = gradient(x)
        x_new = x - alpha(t) * g
        xs.append(x_new)
    return xs

# This generates the plots that appear above
if __name__ == "__main__":
    import os

    import numpy as np
    import pylab as pl
    import plotting

    ### GRADIENT DESCENT ###

    # problem definition

    # the function to be estimate
    function = lambda x : x ** 4
    # its gradient
    gradient = lambda x : 4 * x ** 3 

    step_size = 0.05
    x0 = 1.0
    iters = 10

    # run gradient descent

    iterates = gradient_descent(gradient, x0, lambda x: step_size, iters)

    print iterates

    ### PLOTTING ###
    plotting.plot_iterates_vs_function(iterates, function,
                                     path='figures/iterates.png', y_star=0.0)
    plotting.plot_iteration_vs_function(iterates, function,
                                      path='figures/convergence.png', y_star=0.0)

    # make animation
    try:
        os.makedirs('figures/animation')
    except OSError:
        pass
    for t in range(iters):
        x = iterates[t]
        x_plus = iterates[t+1]

        f = function
        g = gradient
        f_hat = lambda y: f(x) + g(x) * (y - x)

        x_min = (0-f(x))/g(x) + x
        x_max = (1.1-f(x))/g(x) + x

        pl.figure()

        pl.plot(np.linspace(0, 1.1, 100), function(np.linspace(0, 1.1, 100)), alpha=0.2)
        pl.xlim([0, 1.1])
        pl.ylim([0, 1.1])
        pl.xlabel('x')
        pl.ylabel('f(x)')

        pl.plot([x_min, x_max], [f_hat(x_min), f_hat(x_max)], '--', alpha=0.2)
        pl.scatter([x, x_plus], [f(x), f(x_plus)], c=[0.8, 0.2])

        pl.savefig('figures/animation/%02d.png' % t)
        pl.close()
