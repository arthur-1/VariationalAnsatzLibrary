"""
This file implements a variant of the 'RotoSolve' optimization algorithm presented by Ostaszewski et al. in 2019.
arxiv: 1905.09692v1
"""
import numpy as np

class RotoSolve:
    def __init__(self, maxiter):
        self.maxiter = maxiter

    def optimize(self, num_vars, objective_function, initial_point):
        for _ in range(self.maxiter):
            for d in range(num_vars):
                phi = (np.random.rand() * 2 * np.pi) - np.pi
                # phi = initial_point[d]
                initial_point[d] += phi
                exp_phi = objective_function(initial_point)
                initial_point[d] -= phi
                phi_plus = phi + (np.pi / 2)
                initial_point[d] += phi_plus
                exp_phi_plus = objective_function(initial_point)
                initial_point[d] -= phi_plus
                phi_minus = phi - (np.pi / 2)
                initial_point[d] += phi_minus
                exp_phi_minus = objective_function(initial_point)
                initial_point[d] -= phi_minus
                initial_point[d] = -(np.pi / 2) - np.arctan2(2 * exp_phi - exp_phi_plus - exp_phi_minus,
                                                             exp_phi_plus - exp_phi_minus)

        return objective_function(initial_point)


