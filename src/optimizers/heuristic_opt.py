"""
Find the optimal parameterization for a single parameter.

This calculation for a single parameter is based off of: https://arxiv.org/pdf/1905.09692.pdf
"""
import numpy as np
from src.ansatz.ansatz import Ansatz, CircuitCell
from src.optimizers.optimizer import Optimizer
import matplotlib.pyplot as plt

class HeuristicOpt(Optimizer):
    def __init__(self, ansatz: Ansatz):
        super().__init__(ansatz)

    def optimize(self):
        params = self.get_parameter_list()

        raise NotImplementedError


