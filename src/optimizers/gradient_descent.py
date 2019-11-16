import numpy as np

class GradientDescent:
    def __init__(self, maxiter, tol=1e-4):
        self.maxiter = maxiter
        self.eta = 1e-1
        self.termination_thresh = tol

    @staticmethod
    def gradient(i, objective_function, initial_point, eps=1e-8):
        y = (objective_function(initial_point))
        initial_point[i] += eps
        y_prime = (objective_function(initial_point))
        initial_point[i] -= eps
        return (y_prime - y) / eps

    def optimize(self, num_vars, objective_function, initial_point):
        for _ in range(self.maxiter):
            # param_opt = np.random.permutation(list(range(num_vars)))
            # for d in param_opt:
            for d in range(num_vars):
                grad = self.gradient(d, objective_function, initial_point)
                while np.abs(grad) > self.termination_thresh:
                    initial_point[d] -= grad
                    grad = self.gradient(d, objective_function, initial_point)
        return objective_function(initial_point), initial_point
