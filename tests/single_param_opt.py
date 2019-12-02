from src.ansatz.ansatz import Ansatz, AnsatzOperator
from src.qubit_operators.random_hermitian_qubitop import get_qubit_op
from src.optimizers.heuristic_opt import HeuristicOpt
from src.optimizers.generic_opt import GenericOpt
import numpy as np
import matplotlib.pyplot as plt

num_qubits = 2
depth = 2
ansatz_operator = AnsatzOperator(get_qubit_op, num_qubits, backend="qasm_simulator", shots=200, seed=2)
ansatz = Ansatz(ansatz_operator, verbose=True, logging=True)
layer1 = []
layer2 = []
cx_layer = []
for i in range(num_qubits):
    layer1.append(["ry", i, []])
    layer2.append(["rz", i, []])
for i in range(int(num_qubits / 2)):
    cx_layer.append(["cx", i, [i + int(num_qubits / 2)]])

for _ in range(depth):
    ansatz.append_layer(layer1)
    ansatz.append_layer(layer2)
    ansatz.append_layer(cx_layer)

print(ansatz.get_quantum_circuit())


# optimizers = [GenericOpt(ansatz, "COBYLA", maxiter=600)]
# opt_names = ["COBYLA"]
optimizers = [HeuristicOpt(ansatz, budget=100), GenericOpt(ansatz, "SPSA", maxiter=150),
              GenericOpt(ansatz, "COBYLA", maxiter=400)]
opt_names = ["RotoSolve", "SPSA", "COBYLA"]

# optimizers = [HeuristicOpt(ansatz), HeuristicOpt(ansatz)] #, GenericOpt(ansatz, "COBYLA")]
# opt_names = ["Heuristic", "Random"]

for i, optimizer in enumerate(optimizers):
    print("OPT:", i)
    if opt_names[i] == "RotoSolve":
        # ret = optimizer.optimize(False)
        ret = optimizer.optimize(True)
    else:
        ret = optimizer.optimize()
    logs = ansatz.get_logs()
    xs = [x for x in range(len(logs))]
    plt.scatter(xs, logs, label=opt_names[i], s=1.5)
    ansatz.clear_logs()
    ansatz.reset_parameters()

# plt.title("Random " + str(num_qubits) + "-qubits problem.")
plt.xlabel("Circuit Evaluations")
plt.ylabel("Energy Evaluation")
plt.legend()
plt.show()

