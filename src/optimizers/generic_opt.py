"""
Find the optimal parameterization for a single parameter.

This calculation for a single parameter is based off of: https://arxiv.org/pdf/1905.09692.pdf
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


