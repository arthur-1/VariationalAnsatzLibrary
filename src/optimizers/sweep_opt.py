"""
A custom Bayesian parameter optimization system.
The optimal calculation for a single parameter is based off of: https://arxiv.org/pdf/1905.09692.pdf
"""
import numpy as np
from qiskit.aqua.components.optimizers import *
import time

class SweepOpt:
    def __init__(self, maxiter, sweep_count=20, meta_oper_iter=200, seed=234):
        self.maxiter = maxiter
        self.sweep_count = sweep_count
        self.sweepspace_dimension = 4
        self.objective_function = None
        self.X = None
        self.num_vars = None
        # self.optimizer = TNC(maxiter=meta_oper_iter)
        self.optimizer = COBYLA(maxiter=meta_oper_iter, tol=1e-3)
        self.similarity_thresh = 1e-4
        self.seed = seed
        self.initial_point = None

    @staticmethod
    def gram_schmidt_columns(X):
        Q, R = np.linalg.qr(X)
        return Q

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

    def meta_cost_function(self, hyperparams):
        sweeps = []
        for i in range(self.sweepspace_dimension):
            sweeps.append(np.array(self.gram_schmidt_columns(self.X[:,[i]])))
        vec = 0
        for i, hyperparam in enumerate(hyperparams):
            vec += sweeps[i] * hyperparam
        params = [vec[i] for i in range(self.num_vars)]
        self.initial_point = params
        return self.objective_function(params)

    def optimize(self, num_vars, objective_function, initial_point):
        self.objective_function = objective_function
        # self.sweepspace_dimension = int(num_vars / 2)
        self.sweepspace_dimension = num_vars
        self.num_vars = num_vars
        best = objective_function(initial_point)
        best_params = initial_point
        sweep_params = [np.random.rand()] * self.sweepspace_dimension
        for sweep in range(self.sweep_count):
            self.X = np.random.normal(size=(num_vars, self.sweepspace_dimension))
            ret = self.optimizer.optimize(num_vars=self.sweepspace_dimension, objective_function=self.meta_cost_function,
                                          initial_point=np.asarray(sweep_params))
            if ret[1] < best:
                best = ret[1]
            sweep_params = ret[0]

            param_order = list(np.asarray(np.random.permutation(num_vars)))
            final_perm = []
            for _ in range(self.maxiter):
                final_perm += param_order

            for param in final_perm:
                params = self.optimal_single_parameter(objective_function, self.initial_point, param)
                self.initial_point = params
        return [best_params, best]
