import numpy as np
import networkx as nx
# useful additional packages
from qiskit import BasicAer
from qiskit.tools.visualization import plot_histogram
from qiskit.aqua import Operator, run_algorithm
from qiskit.aqua.input import EnergyInput
from qiskit.aqua.translators.ising import max_cut, tsp
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.aqua.components.optimizers import SPSA
import time
from qiskit.aqua.components.variational_forms import RY
from qiskit.aqua import QuantumInstance

offset = 0

def get_qubit_op(num_qubits, seed=23412341234):
    G = nx.Graph()

    # """ HAVE DATA -- VIGO
    n = num_qubits
    SEED = seed
    np.random.seed(SEED % (2 ** 31 - 1))
    G = nx.dense_gnm_random_graph(n, n ** 2, seed=SEED)
    for (u, v, w) in G.edges(data=True):
        w['weight'] = np.random.rand() * 0.5 # 0.1

    np.random.seed(int(time.time()))

    w = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            temp = G.get_edge_data(i, j, default=0)
            if temp != 0:
                w[i, j] = temp['weight']
    # print(w)

    qubitOp, offset = max_cut.get_max_cut_qubitops(w)
    algo_input = EnergyInput(qubitOp)

    pos = nx.spring_layout(G)
    exact = ExactEigensolver(qubitOp).run()
    exact_energy = exact['energy']
    return qubitOp, exact_energy

def decode_result(result, offset=0):
    x = max_cut.sample_most_likely(result['eigvecs'][0])
    print('energy:', result['energy'])
    print('max-cut objective:', result['energy'] + offset)
    print('solution:', max_cut.get_graph_solution(x))
    print('solution objective:', max_cut.max_cut_value(x, w))
    return " "


def validate_solution(x, exac):
    exac = list(reversed(exac))
    op_1 = "".join([str(i) for i in exac])
    op_2 = "".join([str(1 if i == 0 else 0) for i in exac])
    return (x == op_1) or (x == op_2)