from qiskit.aqua.utils import random_hermitian
import numpy as np
import time
from qiskit.aqua import Operator
from qiskit.aqua.algorithms import ExactEigensolver

def get_qubit_op(num_qubits, seed=1):
    np.random.seed(seed)
    H = random_hermitian(2 ** num_qubits)
    qubitOp = Operator(matrix=H)
    cur_time = int(time.time())
    np.random.seed(cur_time)
    exact = ExactEigensolver(qubitOp).run()
    exact_energy = exact['energy']
    return qubitOp, exact_energy

