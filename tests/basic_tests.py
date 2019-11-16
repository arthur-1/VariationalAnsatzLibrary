from src.ansatz.ansatz import Ansatz
from src.qubit_operators.random_hermitian_qubitop import get_qubit_op

num_qubits = 4
qubitOp = get_qubit_op(num_qubits)
ansatz = Ansatz(num_qubits, qubitOp)

ansatz.append_layer([["cu3", 0, [1]],
                     ["cx", 3, [2]],
                     ])

qc = ansatz.get_quantum_circuit()


print(qc)


