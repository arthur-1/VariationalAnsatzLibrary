"""
A custom Bayesian parameter optimization system.
The optimal calculation for a single parameter is based off of: https://arxiv.org/pdf/1905.09692.pdf
"""
import numpy as np
from qiskit.aqua.components.optimizers import *
import time

class CustomBayes:
    def __init__(self, maxiter, random=True, seed=345):
        self.maxiter = maxiter
        self.similarity_thresh = 1e-4
        self.local_opt = COBYLA()
        self.random = random
        self.seed = seed

    def optimal_single_parameter(self, objective_function, initial_point, param_index):
        # Yields the optimal parameterization when ONLY considering param_index (not globally optimal)
        initial_point[param_index] = 0
        exp_0 = objective_function(initial_point)
        initial_point[param_index] = np.pi
        exp_pi = objective_function(initial_point)

        # If the two evaluations are very close to each other, there is no need to waste another 2 function calls
        if np.abs(exp_0 - exp_pi) < self.similarity_thresh:
            return initial_point

        initial_point[param_index] = np.pi / 2
        exp_pi_div_2 = objective_function(initial_point)
        initial_point[param_index] = -np.pi / 2
        exp_minus_pi_div_2 = objective_function(initial_point)

        # If all evaluations where the same, just return.
        if exp_0 == exp_pi == exp_pi_div_2 == exp_minus_pi_div_2:
            return initial_point

        B = np.arctan2(exp_0 - exp_pi, exp_pi_div_2 - exp_minus_pi_div_2)
        # Set the parameter optimally
        theta_opt = -(np.pi / 2) - B + 2 * np.pi
        initial_point[param_index] = theta_opt
        return initial_point

    def optimize(self, num_vars, objective_function, initial_point):
        if self.random:
            np.random.seed(self.seed)
            param_order = list(np.asarray(np.random.permutation(num_vars)))
            param_order = param_order + param_order
            np.random.seed(int(time.time()))

            for param in param_order:
                params = self.optimal_single_parameter(objective_function, initial_point, param)
                initial_point = params
        else:
            print(initial_point)


        return [initial_point, objective_function(initial_point)]
