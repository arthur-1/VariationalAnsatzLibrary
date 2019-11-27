"""
Find the optimal parameterization for a single parameter.

This calculation for a single parameter is based off of: https://arxiv.org/pdf/1905.09692.pdf
"""
import numpy as np
from src.ansatz.ansatz import Ansatz, CircuitCell
from src.optimizers.optimizer import Optimizer
from qiskit.aqua.components.optimizers import COBYLA

# Make a priority node object which inherits CircuitCell


class _Node:
    def __init__(self, cell: CircuitCell):
        self.cell = cell
        self.__priority = 0

    def get_priority(self):
        return self.__priority

    def increment_priority(self):
        self.__priority += 1

    def decrement_priority(self):
        self.__priority -= 1


class _PriorityQueue:
    def __init__(self, cell):
        self.__sentinal = _Node(cell)
        self.__sentinal.increment_priority()
        self.__queue = [self.__sentinal]

    def print(self):
        out = []
        for node in self.__queue:
            out.append((node.cell, node.get_priority()))
        print(out)

    def increment_priority(self, f):
        """
        Increments the priority of the node for which f returns true.
        """
        for node in self.__queue:
            new_n = node.cell.get_qubit()
            new_d = node.cell.get_depth()
            if f(new_n, new_d):
                node.increment_priority()
                break

    def insert(self, new_node):
        self.__queue.append(new_node)
        np.random.shuffle(self.__queue)
        self.__queue.sort(key=lambda x: x.get_priority())

    def pop(self):
        """
        Gets the tail, the node with the highest priority, returns it, and decrements its priority, and removes it
        from the priority queue.
        """
        popped = self.__queue.pop()
        popped.decrement_priority()
        popped.decrement_priority()
        popped.decrement_priority()

        # if np.random.rand() < 0.5:
        #     popped.decrement_priority()
        # if np.random.rand() < 0.5:
        #     popped.decrement_priority()
        # if np.random.rand() < 0.5:
        #     popped.decrement_priority()
        return popped


class HeuristicOpt(Optimizer):
    def __init__(self, ansatz: Ansatz, budget=120):
        super().__init__(ansatz)
        self.__similarity_thresh = 1e-2
        self.sweep_count = 1
        self.maxiter = 200
        self.budget = budget
        self.sweep_divisions = 4
        self.optimizer = COBYLA(maxiter=self.maxiter, tol=1e-3)
        self.objective_function = None
        self.sweepspace_dimension = None
        self.num_vars = None

    def optimal_single_parameter(self, param_index):
        objective_function = self.ansatz.cost_function
        initial_point = self.get_parameter_list()
        # Yields the optimal parameterization when ONLY considering param_index (not globally optimal)
        initial_point[param_index] = 0
        self.set_parameter_list(initial_point)
        exp_0 = objective_function()
        initial_point[param_index] = np.pi
        self.set_parameter_list(initial_point)
        exp_pi = objective_function()

        # If the two evaluations are very close to each other, there is no need to waste another 2 function calls
        if np.abs(exp_0 - exp_pi) < self.__similarity_thresh:
            self.set_parameter_list(initial_point)
            return

        initial_point[param_index] = np.pi / 2
        self.set_parameter_list(initial_point)
        exp_pi_div_2 = objective_function()
        initial_point[param_index] = -np.pi / 2
        self.set_parameter_list(initial_point)
        exp_minus_pi_div_2 = objective_function()

        # If all evaluations where the same, just return.
        if exp_0 == exp_pi == exp_pi_div_2 == exp_minus_pi_div_2:
            self.set_parameter_list()
            return

        B = np.arctan2(exp_0 - exp_pi, exp_pi_div_2 - exp_minus_pi_div_2)
        # Set the parameter optimally
        theta_opt = -(np.pi / 2) - B + 2 * np.pi
        initial_point[param_index] = theta_opt
        self.set_parameter_list(initial_point)
        return

    def get_parameter_index(self, best_node):
            targ_q = best_node.cell.get_qubit()
            targ_d = best_node.cell.get_depth()
            # Get the index in the parameter array of the node to optimize
            index = 0
            for ii in range(self.max_depth):
                for i in range(self.num_qubits):
                    if (i == targ_q) and (ii == targ_d):
                        return index - 1
                    else:
                        index += self.ansatz[i, ii].get_param_count()
            raise AssertionError("[ASSERTION FAILURE] Unreachable code reached.")

    @staticmethod
    def gram_schmidt_columns(X):
        Q, R = np.linalg.qr(X)
        return Q

    def meta_cost_function(self, hyperparams):
        sweeps = []
        for i in range(self.sweepspace_dimension):
            sweeps.append(np.array(self.gram_schmidt_columns(self.X[:,[i]])))
        vec = 0
        for i, hyperparam in enumerate(hyperparams):
            vec += sweeps[i] * hyperparam
        params = [vec[i] for i in range(self.num_vars)]
        self.set_parameter_list(params)
        return self.objective_function()

    def sweep(self):
        initial_point = self.get_parameter_list()
        num_vars = len(initial_point)
        self.sweepspace_dimension = int(num_vars / self.sweep_divisions)
        self.num_vars = num_vars
        best = self.objective_function()
        sweep_params = [np.random.rand()] * self.sweepspace_dimension
        best_params = sweep_params
        for sweep in range(self.sweep_count):
            self.X = np.random.normal(size=(num_vars, self.sweepspace_dimension))
            ret = self.optimizer.optimize(num_vars=self.sweepspace_dimension, objective_function=self.meta_cost_function,
                                          initial_point=np.asarray(sweep_params))
            sweep_params = ret[0]
            if ret[1] < best:
                best = ret[1]
                best_params = sweep_params

        self.meta_cost_function(best_params)

    def optimize(self, random=False):
        # params = self.get_parameter_list()
        budget = self.budget

        if random:
            # permutation = list(np.random.permutation([i for i in range(len(self.get_parameter_list()))]))
            permutation = [i for i in range(len(self.get_parameter_list()))]
            permutation *= int(budget / len(permutation))
            for i in permutation:
                self.optimal_single_parameter(i)
        else:
            target_n = self.num_qubits - 1
            target_d = 0
            # target_n = np.random.randint(0, self.num_qubits)
            # target_d = np.random.randint(0, self.max_depth)
            # @TODO Run sweep opt on the parameters on the deepest layer
            # Pick a random node to add to the priority queue first.
            self.objective_function = self.ansatz.cost_function
            self.sweep()

            first_cell = self.ansatz[target_n, target_d]
            pq = _PriorityQueue(first_cell)

            # Iterate over all nodes in the ansatz, and add each to the PQ, if it is not the node that was already selected
            for i in range(self.num_qubits):
                for ii in range(self.max_depth):
                    if (i != target_n) or (ii != target_d):
                        new_item = _Node(self.ansatz[i, ii])
                        pq.insert(new_item)


            for i in range(budget):
                best_node = pq.pop()
                next_cell = best_node.cell.get_next_cell()

                if next_cell is not None:
                    next_cell_n = next_cell.get_qubit()
                    next_cell_d = next_cell.get_depth()
                    f = lambda xn, xd: (next_cell_n == xn) and (next_cell_d == xd)
                    pq.increment_priority(f)
                bd = best_node.cell.get_depth()
                bn = best_node.cell.get_qubit()
                if bn + 1 < self.num_qubits:
                    f = lambda xn, xd: (xn == bn + 1) and (xd == bd)
                    pq.increment_priority(f)
                if bd + 1 < self.max_depth:
                    f = lambda xn, xd: (xn == bn) and (xd == bd + 1)
                    pq.increment_priority(f)
                if bd - 1 >= 0:
                    f = lambda xn, xd: (xn == bn) and (xd == bd - 1)
                    pq.increment_priority(f)

                param_index = self.get_parameter_index(best_node)
                self.optimal_single_parameter(param_index)
                pq.insert(best_node)

                if np.random.rand() < 0.005:
                    self.sweep()


        return


