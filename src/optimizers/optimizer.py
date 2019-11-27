from src.ansatz.ansatz import Ansatz, CircuitCell


class Optimizer:
    def __init__(self, ansatz: Ansatz):
        self.ansatz = ansatz
        self.num_qubits = self.ansatz.get_num_qubits()
        self.max_depth = self.ansatz.get_depth()

    def get_parameter_list(self):
        params = []
        for ii in range(self.max_depth):
            for i in range(self.num_qubits):
                ret = self.ansatz[i, ii]
                if ret.has_parameters():
                    params += ret.get_parameters()
        return params

    def set_parameter_list(self, params):
        for ii in range(self.max_depth):
            for i in range(self.num_qubits):
                ret = self.ansatz[i, ii]
                num_params = ret.get_param_count()
                if num_params > 0:
                    new_params = params[:num_params]
                    ret.set_parameters(new_params)
                    params = params[num_params:]


