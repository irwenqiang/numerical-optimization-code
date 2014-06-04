# This generates the plots that appear above
import  ..yannopt/yannopt/optimizers/lbfgs.py

'''
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
'''
