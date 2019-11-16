"""
This file implements a Bayesian parameter optimization scheme for the parameters of fixed variational circuits, as
described in https://pdfs.semanticscholar.org/c5f1/7d67dc1decdf73cce890e1a25a9062e68f28.pdf#cite.vqe2

This specific implementation uses code from: https://thuijskens.github.io/2016/12/29/bayesian-optimisation/
"""
import numpy as np
# import sklearn.gaussian_process as gp
from sklearn.base import clone
from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern


class NaiveBayesian:
    def __init__(self, maxiter, xs=None, ys=None, noise=0):
        self.maxiter = maxiter
        self.noise = noise

        if xs is None:
            self.xs = []
        else:
            self.xs = xs
        if ys is None:
            self.ys = []
        else:
            self.ys = ys

    def optimize(self, num_vars, objective_function, initial_point):
        # Use custom kernel and estimator to match previous example
        m52 = ConstantKernel(0.5) * Matern(length_scale=1, nu=1.5)
        gpr = GaussianProcessRegressor(kernel=m52, alpha=self.noise ** 2)

        # x_new = initial_point
        # y_new = objective_function(initial_point)
        # self.xs.append(x_new)
        # self.ys.append(y_new)
        # print(x_new)
        # print(y_new)
        dims = [(0, 2 * np.pi)] * num_vars
        r = gp_minimize(lambda x: objective_function(x),
                        dimensions=dims,
                        base_estimator=gpr,
                        acq_func='EI',  # expected improvement
                        # acq_optimizer='sampling',
                        xi=1,  # exploitation-exploration trade-off
                        n_calls=self.maxiter,  # number of iterations
                        n_random_starts=10,  # initial samples are provided
                        x0=initial_point,  # initial samples
                        )

        print(r.x_iters, r.func_vals)
        # Fit GP model to samples for plotting results
        ret = gpr.fit(r.x_iters, r.func_vals)

        # gpr.
        print(ret)


        return ret
