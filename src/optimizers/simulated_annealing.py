from scipy.optimize import basinhopping

class SimulatedAnnealing:
    def __init__(self, maxiter, method="COBYLA"):
        self.maxiter = maxiter
        self.method = method

    def optimize(self, num_vars, objective_function, initial_point):
        minimizer_kwargs = {"method": self.method}
        ret = basinhopping(objective_function, initial_point, minimizer_kwargs=minimizer_kwargs, niter=self.maxiter)
        initial_point = ret.x
        return [initial_point, objective_function(initial_point)]
