"""
This is a wrapper enabling compatibility with arbitrary off-the-shelf optimizers.
"""
import numpy as np
from src.ansatz.ansatz import Ansatz, CircuitCell
from qiskit.aqua.components.optimizers import *
from src.optimizers.optimizer import Optimizer


class GenericOpt(Optimizer):
    def __init__(self, ansatz: Ansatz, opt_name: str="COBYLA", maxiter: int=1000, tol: float=1e-4):
        super().__init__(ansatz)
        if opt_name == "COBYLA":
            self.optimizer = COBYLA(maxiter=maxiter, tol=tol)
        elif opt_name == "SLSQP":
            self.optimizer = SLSQP(maxiter=maxiter, tol=tol)
        elif opt_name == "SPSA":
            # self.optimizer = SPSA(max_trials=maxiter, c0=0.05, c1=0.05, c2=0.3, c3=0.2)
            self.optimizer = SPSA(max_trials=maxiter, c0=0.05, c1=0.05, )
            # self.optimizer = SPSA(max_trials=maxiter)
        else:
            raise AssertionError("[ASSERTION FAILURE] Unrecognized optimizer.")

    def cost_function(self, params):
        self.set_parameter_list(params)
        cost = self.ansatz.cost_function()
        return cost

    def optimize(self):
        init_params = self.get_parameter_list()
        ret = self.optimizer.optimize(len(init_params), objective_function=self.cost_function,
                                      initial_point=init_params)
        return ret


