from qiskit.aqua.utils import random_hermitian
import numpy as np
import time
from qiskit.aqua import Operator


def get_qubit_op(num_qubits, backend, seed=1):
    np.random.seed(seed)
    H = random_hermitian(2 ** num_qubits)
    qubitOp = Operator(matrix=H)
    cur_time = int(time.time())
    np.random.seed(cur_time)
    return qubitOp

