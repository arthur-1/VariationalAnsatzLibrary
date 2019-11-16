"""
Find the optimal parameterization for a single parameter.

This calculation for a single parameter is based off of: https://arxiv.org/pdf/1905.09692.pdf
"""
import numpy as np
import matplotlib.pyplot as plt

class ArthurSolve:
    def __init__(self, maxiter):
        self.maxiter = maxiter

    def optimize(self, num_vars, objective_function, initial_point):
        initial_point[0] = 0
        exp_0 = objective_function(initial_point)
        initial_point[0] = np.pi
        exp_pi = objective_function(initial_point)
        initial_point[0] = np.pi / 2
        exp_pi_div_2 = objective_function(initial_point)
        initial_point[0] = -np.pi / 2
        exp_minus_pi_div_2 = objective_function(initial_point)

        # A = (1 / 2) * np.sqrt((exp_0 -exp_pi) ** 2 + (exp_pi_div_2 - exp_minus_pi_div_2) ** 2)
        B = np.arctan2(exp_0 - exp_pi, exp_pi_div_2 - exp_minus_pi_div_2)
        # C = (1 / 2) * (exp_0 + exp_pi)

        # Set the parameter optimally
        theta_opt = -(np.pi / 2) - B + 2 * np.pi
        initial_point[0] = theta_opt

        return [initial_point, objective_function(initial_point)]

