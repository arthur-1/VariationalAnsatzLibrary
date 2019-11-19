from src.ansatz.ansatz import Ansatz
from src.qubit_operators.random_hermitian_qubitop import get_qubit_op
from src.optimizers.heuristic_opt import HeuristicOpt
from src.optimizers.generic_opt import GenericOpt
import numpy as np

num_qubits = 6

ansatz = Ansatz(num_qubits, get_qubit_op)

# ansatz.append_layer([["cu3", 0, [1]],
#                      ["cu3", 3, [2]],
#                      ["u3", 4, []],
#                      ["rx", 5, []]
#                      ])
# ansatz.append_layer([["cu3", 1, [3]],
#                      ["cu3", 5, [4]],
#                      ["u3", 2, []],
#                      ["ry", 0, []]
#                      ])
ansatz.append_layer([["u3", 0, []],
                     ["u3", 1, []],
                     ["u3", 2, []],
                     ["u3", 3, []],
                     ["u3", 4, []],
                     ["u3", 5, []],
                     ])

print(ansatz.get_quantum_circuit())


optimizer = GenericOpt(ansatz)
ret = optimizer.optimize()

# qc = ansatz.get(params)

