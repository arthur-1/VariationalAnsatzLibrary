from src.exceptions.exceptions import *
import numpy as np
from qiskit import Aer, BasicAer
from enum import IntEnum
from qiskit.aqua import QuantumInstance
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


class CircuitCell:
    def __init__(self, gate_type, is_target: bool, qubit_index: int=None, depth_index: int=None, parameters=None):
        """
        :param gate_type: What type the gate is. Currently, controls are only supported as identity gates.
        :param is_target: Any gate which is not a control, is a target. I.e. a target may have 0 or more controls.
        """
        self.__is_target = is_target
        self.__gate_type = _GateType.get_gate_type_from_string(gate_type)
        if parameters is None:
            self.__parameters = []
        else:
            assert 0 <= len(parameters) <= 3, "[ASSERTION FAILURE] Gates with more than three parameters are not" \
                                              + "supported."
            self.__parameters = parameters

        if qubit_index is not None and depth_index is not None:
            self.__qubit = qubit_index
            self.__depth = depth_index
        elif (qubit_index is None) != (depth_index is None):
            raise AssertionError("[ASSERTION FAILURE] If layer or qubit is specified, the qubit or layer must also be"
                                 + "specified. Cannot specify one without the other.")
        else:
            self.__qubit = self.__depth = None
        # The next target or control. If this is a target, this should always be None.
        self.__next_cell = None
        # The previous control, noting that a previous cell can never be a target.
        self.__prev_cell = None

    def has_parameters(self):
        return self.get_param_count() > 0

    def get_param_count(self):
        return len(self.__parameters)

    def get_parameters(self):
        return self.__parameters

    def set_parameters(self, parameters):
        assert 0 <= len(parameters) <= 3, "[ASSERTION FAILURE] Gates with more than three parameters are not" \
                                          + "supported."
        gate_type_str = _GateType.get_string_from_gate_type(self.__gate_type)
        gate_type_param_req = _GateType.get_gate_type_param_count(gate_type_str)
        assert gate_type_param_req == len(parameters), "[ASSERTION FAILURE] Gate type (" + gate_type_str \
                                                       + ") requires " + str(gate_type_param_req) \
                                                       + " parameters instead of " + str(len(parameters)) + "."
        self.__parameters = parameters

    def is_target(self):
        return self.__is_target

    def get_qubit(self):
        return self.__qubit

    def get_depth(self):
        return self.__depth

    def get_next_cell(self):
        return self.__next_cell

    def get_prev_cell(self):
        return self.__prev_cell

    def get_gate_name(self):
        return _GateType.get_string_from_gate_type(self._get_type())

    def _get_type(self):
        return self.__gate_type

    def _set_next_cell(self, cell):
        # This cell cannot link to another cell if this is a target, as targets are assumed to be terminal.
        assert not self.__is_target, "[ASSERTION FAILURE] Cannot set a linked cell for a terminal target gate."
        assert cell is not None, "[ASSERTION FAILURE] Cannot set next cell to None, is None by default."
        self.__next_cell = cell

    def _set_prev_cell(self, cell):
        self.__prev_cell = cell


class AnsatzOperator:
    def __init__(self, get_operator, num_qubits: int, backend: str="statevector_simulator",
                 shots: int=1024, seed: int=1):
        self.__num_qubits = num_qubits
        self.__operator = get_operator(num_qubits, seed)
        self.__backend_str = backend
        if backend == "statevector_simulator":
            self.__backend = Aer.get_backend(backend)
            self.__mode = "matrix"
            self.__quantum_instance = QuantumInstance(backend=self.__backend)
        elif backend == "qasm_simulator":
            self.__backend = Aer.get_backend(backend)
            self.__mode = "grouped_paulis"
            self.__quantum_instance = QuantumInstance(backend=self.__backend, shots=shots,
                                                      skip_qobj_deepcopy=True
                                                      )
        else:
            raise UnrecognizedInputError

    def _get_num_qubits(self):
        return self.__num_qubits

    def _get_backend(self):
        return self.__backend

    def _get_mode(self):
        return self.__mode

    def _get_operator(self):
        return self.__operator

    def _get_backend_str(self):
        return self.__backend_str

    def _get_quantum_instance(self):
        return self.__quantum_instance


class Ansatz:
    def __init__(self, ansatz_operator: AnsatzOperator, logging: bool=False, verbose: bool=False):
        self.__num_qubits = ansatz_operator._get_num_qubits()
        self.__quantum_instance = ansatz_operator._get_quantum_instance()
        self.__verbose = verbose
        self.__logging_enabled = logging
        self.__log = []
        self.__qubit_operator = ansatz_operator._get_operator()
        self.__backend = ansatz_operator._get_backend()
        backend_str = ansatz_operator._get_backend_str()
        self.__mode = ansatz_operator._get_mode()
        self.__abstract_quantum_circuit = _QuantumCircuitRepresentation(self.__num_qubits, backend_str)

    def __getitem__(self, key) -> CircuitCell:
        """
        Accepts a key, indexed by [qubit, depth], and returns the CircuitCell object at the specified location.
        """
        assert len(key) == 2, "[ASSERTION FAILURE] Attempted to index into an Ansatz object without specifying" \
                              + "depth and qubit."
        qubit = key[0]
        depth = key[1]
        cell = self.__abstract_quantum_circuit.get_quantum_cell(depth, qubit)
        return cell

    def get_num_qubits(self):
        return self.__num_qubits

    def get_depth(self):
        return self.__abstract_quantum_circuit.get_depth()

    def get_quantum_circuit(self):
        return self.__abstract_quantum_circuit.get_quantum_circuit()

    def cost_function(self):
        """
        Even though this method directly calls the private cost function method, this additional method has been added
        to enable a planned change without breaking the API.
        :return: The cost evaluation of the Ansatz with its current parameterization.
        """
        cost = self.__cost_function()
        if self.__verbose:
            print(cost)
        if self.__logging_enabled:
            self.__log.append(cost)
        return cost

    def clear_logs(self):
        self.__log = []

    def reset_parameters(self):
        _n = self.get_num_qubits()
        _d = self.get_depth()
        for i in range(_n):
            for ii in range(_d):
                cell = self.__abstract_quantum_circuit.get_quantum_cell(ii, i)
                params = cell.get_parameters()
                new_params = [0] * len(params)
                cell.set_parameters(new_params)

    def get_logs(self):
        return self.__log

    def append_layer(self, layer):
        """
        Appends the layer of the specified format.
        :param layer: layer must be of the format, [["gate_name: string, target: int, control_list: list, params: list],...]
        """
        self.__abstract_quantum_circuit.append_layer(layer)

    def __cost_function(self):
        """
        Implements the expectation value of the Ansatz over the quantum instance assigned to this object.
        Does not handle setting the parameters of the Ansatz.
        """
        qc = self.get_quantum_circuit()
        qc = self.__qubit_operator.construct_evaluation_circuit(self.__mode, qc, self.__backend)
        result = self.__quantum_instance.execute(qc)
        mean, std = self.__qubit_operator.evaluate_with_result(self.__mode, qc, self.__backend, result)
        mean = np.real(mean)
        return mean


class _GateType(IntEnum):
    U3 = 0
    IDENTITY = 1
    H = 2
    X = 3
    Y = 4
    Z = 5
    S = 6
    T = 7
    RX = 8
    RY = 9
    RZ = 10
    # SPECIAL GATE TYPES, ONLY USED TO DIFFERENTIATE SINGLE QUBIT AND MULTIQUBIT GATES
    CX = -1
    CU3 = -2

    @staticmethod
    def get_gate_type_from_string(gate_type):
        # TODO Remove temporary hack with if statements
        if gate_type == "cx":
            return _GateType.CX
        if gate_type == "cu3":
            return _GateType.CU3
        for i, string in enumerate(_GateType.__converter()):
            if string == gate_type:
                return _GateType(i)
        raise UnrecognizedGateType

    @staticmethod
    def get_string_from_gate_type(gate_type):
        conv = _GateType.__converter()
        gate_type = int(gate_type)
        if 0 <= gate_type < len(conv):
            return conv[gate_type]
        raise UnrecognizedGateType

    @staticmethod
    def get_gate_qubit_rank(gate_type: str):
        formal_type = _GateType.get_gate_type_from_string(gate_type)
        if (formal_type == _GateType.CX) or (formal_type == _GateType.CU3):
            return 2
        else:  # Unrecognized gate types would have been caught in call to _GateType.get_gate_type_from_string
            return 1

    @staticmethod
    def get_target_type_as_string(gate_type: str):
        if gate_type == "cx":
            return "x"
        elif gate_type == "cu3":
            return "u3"
        else:
            raise UnrecognizedGateType

    @staticmethod
    def get_gate_type_param_count(gate_type: str):
        gate_type = _GateType.get_gate_type_from_string(gate_type)
        if (gate_type == _GateType.U3) or (gate_type == _GateType.CU3):
            return 3
        elif (gate_type == _GateType.RX) or (gate_type == _GateType.RY) or (gate_type == _GateType.RZ):
            return 1
        else:
            return 0

    @staticmethod
    def __converter():
        converter = ("u3", "id", "h", "x", "y", "z", "s", "t", "rx", "ry", "rz")
        return converter


class _QuantumCircuitRepresentation:
    def __init__(self, num_qubits, backend: str):
        self.__num_qubits = num_qubits
        # self.__circuit is of the format [[[_CircuitCell], ...], ...]]
        # where self.__circuit[depth][qubit] is the access pattern
        self.__circuit = []
        self.__qr = QuantumRegister(self.__num_qubits, "q")
        self.__backend = backend

    def get_quantum_cell(self, depth, qubit):
        min_list_len = len(min(self.__circuit, key=lambda x: len(x)))
        assert (min_list_len != 0), "[ASSERTION FAILURE] Attempted to index an ansatz with no layers added."
        assert 0 <= depth <= self.get_depth() - 1, "[ASSERTION FAILURE] Specified circuit layer (" + str(depth) \
                                                      + ") is invalid."
        assert 0 <= qubit <= self.__num_qubits - 1, "[ASSERTION FAILURE] Specified qubit index (" + str(qubit) \
                                                         + ") is invalid."
        return self.__circuit[depth][qubit]

    def get_depth(self):
        """
        Returns the depth of the quantum circuit object.
        """
        return len(self.__circuit)

    def get_quantum_circuit(self):
        qc = QuantumCircuit(self.__qr)

        for depth, layer in enumerate(self.__circuit):
            for qubit, cell in enumerate(layer):
                is_cell_target = cell.is_target()
                cell_type = cell._get_type()
                if (not is_cell_target) and (cell_type != _GateType.IDENTITY):
                    raise AssertionError("[ASSERTION FAILURE] Control must be an identity gate.")
                controls = []
                # If the cell is a target, find all of its controls
                if is_cell_target:
                    prev_cell = cell.get_prev_cell()
                    while prev_cell is not None:
                        controls.append(prev_cell)
                        prev_cell = prev_cell.get_prev_cell()
                    # Sort controls by qubit_id
                    controls.sort(key=lambda c: c.get_qubit())
                    assert len(controls) <= 1, "[ASSERTION FAILURE] Gates with more than one control are not currently"\
                                               + "supported."
                    # @TODO Add support for arbitrary controlled-U gate by setting parameters of a controlled-U3
                    if len(controls) == 1:
                        if cell_type == _GateType.X:
                            qc.cx(self.__qr[controls[0].get_qubit()], self.__qr[qubit])
                        elif cell_type == _GateType.U3:
                            params = cell.get_parameters()
                            assert len(params) == 3, "[ASSERTION FAILURE] CU3 gate has the incorrect number of " +\
                                                     "parameters specified."
                            qc.cu3(params[0], params[1], params[2], self.__qr[controls[0].get_qubit()], self.__qr[qubit])
                    elif len(controls) == 0:
                        params = cell.get_parameters()
                        if cell_type == _GateType.U3:
                            assert len(params) == 3, "[ASSERTION FAILURE] Must give three parameters for a U3 gate."
                            qc.u3(params[0], params[1], params[2], qubit)
                        elif cell_type == _GateType.IDENTITY:
                            qc.iden(qubit)
                        elif cell_type == _GateType.H:
                            qc.h(qubit)
                        elif cell_type == _GateType.X:
                            qc.x(qubit)
                        elif cell_type == _GateType.Y:
                            qc.y(qubit)
                        elif cell_type == _GateType.Z:
                            qc.z(qubit)
                        elif cell_type == _GateType.S:
                            qc.s(qubit)
                        elif cell_type == _GateType.T:
                            qc.t(qubit)
                        elif cell_type == _GateType.RX:
                            assert len(params) == 1, "[ASSERTION FAILURE] Must give one parameter for RX gate."
                            qc.rx(params[0], qubit)
                        elif cell_type == _GateType.RY:
                            assert len(params) == 1, "[ASSERTION FAILURE] Must give one parameter for RY gate."
                            qc.ry(params[0], qubit)
                        elif cell_type == _GateType.RZ:
                            assert len(params) == 1, "[ASSERTION FAILURE] Must give one parameter for RZ gate."
                            qc.rz(params[0], qubit)
                    else:
                        raise UnknownError
        return qc

    def append_layer(self, layer: list):
        # Sort the list in ascending order of control list length
        current_depth = len(self.__circuit)
        assert min([len(l) for l in layer]) >= 3, "[ASSERTION FAILURE] Insufficient arguments given for gate in layer."
        layer.sort(key=lambda abstract_cell: (len(abstract_cell[2]), abstract_cell[1]))
        new_layer = [None for _ in range(self.__num_qubits)]
        for abstract_cell in layer:
            assert 5 > len(abstract_cell) >= 3, "[ASSERTION FAILURE] Gate list contains gate with incorrect format."
            cell_label = abstract_cell[0]
            assert isinstance(cell_label, str), "[ASSERTION FAILURE] Cell label must be a string."
            cell_target = abstract_cell[1]
            assert isinstance(cell_target, int), "[ASSERTION FAILURE] Cell target must be an int."
            cell_controls = abstract_cell[2]
            assert isinstance(cell_controls, list), "[ASSERTION FAILURE] Cell controls must be a list."
            if (len(abstract_cell) == 4) and (len(cell_controls) == _GateType.get_gate_type_param_count(cell_label)):
                cell_params = abstract_cell[3]
                assert len(cell_params) == _GateType.get_gate_type_param_count(cell_label), \
                    "[ASSERTION FAILURE] Incorrect number of parameters given for " + cell_label + " gate."
            else:
                cell_params = [0] * _GateType.get_gate_type_param_count(cell_label)
            assert isinstance(cell_params, list), "[ASSERTION FAILURE] Cell params must be a list."
            cell_rank = _GateType.get_gate_qubit_rank(cell_label)
            if cell_rank == 1:
                # All single qubit gates are implicitly their own targets.
                new_cell = CircuitCell(gate_type=cell_label, is_target=True, qubit_index=cell_target,
                                        depth_index=current_depth, parameters=cell_params)
                new_layer[cell_target] = new_cell
            elif cell_rank == 2:
                # Must create each of the control qubits
                control_list = []
                for control in cell_controls:
                    # NOTE CONTROL CELLS CURRENTLY ONLY SUPPORT IDENTITY CONTROLS
                    control_cell = CircuitCell(gate_type='id', is_target=False, qubit_index=control,
                                                depth_index=current_depth)
                    new_layer[control] = control_cell
                    control_list.append(control_cell)
                assert len(control_list) >= 1, "[ASSERTION FAILURE] Gate with rank of at least 2 failed to generate " \
                                               + "any control cells."
                # Now create the target cell
                target_cell_type = _GateType.get_target_type_as_string(cell_label)
                target_cell = CircuitCell(gate_type=target_cell_type, is_target=True, qubit_index=cell_target,
                                           depth_index=current_depth, parameters=cell_params)
                control_cell = control_list.pop()
                target_cell._set_prev_cell(control_cell)
                control_cell._set_next_cell(target_cell)
                for control_cell_prime in control_list:
                    control_cell_prime._set_next_cell(control_cell)
                    control_cell._set_prev_cell(control_cell_prime)
                    control_cell = control_cell_prime
                new_layer[cell_target] = target_cell
            else:
                raise AssertionError("[ASSERTION FAILURE] Gates with more than one control are not currently supported.")
        assert not (None in new_layer), "[ASSERTION_FAILURE] Insufficient gates given in appended layer."
        self.__circuit.append(new_layer)

    def __set_specified_layer_parameters(self):
        raise NotImplementedError

    def __get_abstract_circuit(self):
        return self.__circuit


