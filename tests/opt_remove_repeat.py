from src.ansatz.ansatz import Ansatz, AnsatzOperator
from src.qubit_operators.random_hermitian_qubitop import get_qubit_op
from src.optimizers.heuristic_opt import HeuristicOpt
from src.optimizers.generic_opt import GenericOpt
import numpy as np
import matplotlib.pyplot as plt

lower = -np.pi
upper = np.pi

STEP_SIZE = 40
num_qubits = 2
ansatz_operator = AnsatzOperator(get_qubit_op, num_qubits, backend="statevector_simulator", seed=2)
ansatz = Ansatz(ansatz_operator, verbose=True, logging=True)
layer1 = []
cx_layer = []
for i in range(num_qubits):
    layer1.append(["ry", i, []])
for i in range(int(num_qubits / 2)):
    cx_layer.append(["cx", i, [i + int(num_qubits / 2)]])

ansatz.append_layer(layer1)
ansatz.append_layer(cx_layer)

print(ansatz.get_quantum_circuit())

xs = []
ys = []
zs = []
for theta1 in np.linspace(lower, upper, STEP_SIZE):
    for theta2 in np.linspace(lower, upper, STEP_SIZE):
        ansatz[0, 0].set_parameters([theta1])
        ansatz[1, 0].set_parameters([theta2])
        xs.append(theta1)
        ys.append(theta2)
        zs.append(ansatz.cost_function())

print(xs)
print(ys)
print(zs)


x = np.unique(np.asarray(xs))
y = np.unique(np.asarray(ys))
X, Y = np.meshgrid(x, y)

Z = np.asarray(zs).reshape(len(y), len(x))

plt.pcolormesh(X, Y, Z)
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")

plt.show()


