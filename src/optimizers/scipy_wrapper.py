import numpy as np
from scipy.optimize import minimize

class ScipyWrapper:
    def __init__(self, maxiter, method="BFGS"):
        self.maxiter = maxiter
        self.eta = 1e-1
        self.method = method

    def optimize(self, num_vars, objective_function, initial_point):
        res = minimize(objective_function, initial_point, method=self.method,
                       options = {'gtol': 1e-6, 'disp': True})
        initial_point = res.x
        return [initial_point, objective_function(initial_point)]
