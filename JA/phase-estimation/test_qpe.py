import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import unittest
import numpy as np
from qulacs import QuantumState, QuantumCircuit
from qulacs.state import inner_product

class QPETestClass(unittest.TestCase):
    def test_CU_Z0(self):
        n_qubits = 2
        a_idx = 1
        theta = np.pi/8

        state = QuantumState(n_qubits)
        input_states_bin = [0b00, 0b10]
        input_states = []
        output_states = []

        circuit_H = QuantumCircuit(n_qubits)
        circuit_H.add_H_gate(0)
        # |0>|+> and |1>|+>
        for b in input_states_bin:
            psi = state.copy()
            psi.set_computational_basis(b) 
            input_states += [psi]
            circuit_H.update_quantum_state(psi)

        circuit = QuantumCircuit(n_qubits)
        circuit.add_RZ_gate(0, -0.5*theta)
        circuit.add_CNOT_gate(a_idx, 0)
        circuit.add_RZ_gate(0, 0.5*theta)
        circuit.add_CNOT_gate(a_idx, 0)
        
        for in_state in input_states:
            psi = in_state.copy()
            circuit.update_quantum_state(psi)
            output_states += [psi]

        p_list = []
        for in_state in input_states:
            for out_state in output_states:
                p_list += [inner_product(in_state, out_state)]
        
        # <0|<+|0>|+> = 1
        # <0|<+|1>|H> = 0
        # <1|<+|0>|+> = 0
        # <1|<+|1>|H> = cos(pi/16)
        exp_list = [1.0, 0.0, 0.0, np.cos(theta/2)]

        for result, expected in zip(p_list, exp_list):
            self.assertAlmostEqual(result, expected, places=6)
            # print(result, expected)

    def test_CU_X0X1(self):
        n_qubits = 3
        a_idx = 2
        theta = np.pi/4
        state = QuantumState(n_qubits)
        input_states_bin = [0b001, 0b010, 0b101, 0b110]
        input_states = []
        output_states = []

        circuit = QuantumCircuit(n_qubits)
        # change basis from Z to X
        circuit.add_H_gate(0)
        circuit.add_H_gate(1)
        circuit.add_CNOT_gate(1, 0)
        # RZ
        circuit.add_RZ_gate(0, -0.5*theta)
        circuit.add_CNOT_gate(a_idx, 0)
        circuit.add_RZ_gate(0, 0.5*theta)
        circuit.add_CNOT_gate(a_idx, 0)

        circuit.add_CNOT_gate(1, 0)
        # change basis from Z to X
        circuit.add_H_gate(0)
        circuit.add_H_gate(1)

        for b in input_states_bin:
            psi = state.copy()
            psi.set_computational_basis(b) 
            input_states += [psi]
            psi_out = psi.copy()
            circuit.update_quantum_state(psi_out)
            output_states += [psi_out]

        p_list = []
        for in_state in input_states:
            for out_state in output_states:
                prod = inner_product(in_state, out_state)
                p_list += [prod]
        # |001>
        exp_list = [1.0, 0.0, 0.0, 0.0]
        # |010>
        exp_list += [0.0, 1.0, 0.0, 0.0]
        # |101>
        exp_list += [0.0, 0.0, np.cos(theta/2), complex(0, -np.sin(theta/2))]
        # |110> = (1+cos(pi/8))/2
        exp_list += [0.0, 0.0, complex(0, -np.sin(theta/2)), np.cos(theta/2)]
        
        for result, expected in zip(p_list, exp_list):
            self.assertAlmostEqual(result, expected, places=6)
            # print(result, expected)

    def test_CU_Y0Y1(self):
        n_qubits = 3
        a_idx = 2
        theta = np.pi/4
        state = QuantumState(n_qubits)
        input_states_bin = [0b001, 0b010, 0b101, 0b110]
        input_states = []
        output_states = []

        circuit = QuantumCircuit(n_qubits)
        # change basis from Z to Y
        circuit.add_S_gate(0)
        circuit.add_S_gate(1)
        circuit.add_H_gate(0)
        circuit.add_H_gate(1)
        circuit.add_CNOT_gate(1, 0)
        # RZ
        circuit.add_RZ_gate(0, -0.5*theta)
        circuit.add_CNOT_gate(a_idx, 0)
        circuit.add_RZ_gate(0, 0.5*theta)
        circuit.add_CNOT_gate(a_idx, 0)
        
        circuit.add_CNOT_gate(1, 0)
        # change basis from Z to Y
        circuit.add_H_gate(0)
        circuit.add_H_gate(1)
        circuit.add_Sdag_gate(0)
        circuit.add_Sdag_gate(1)

        for b in input_states_bin:
            psi = state.copy()
            psi.set_computational_basis(b) 
            input_states += [psi]
            psi_out = psi.copy()
            circuit.update_quantum_state(psi_out)
            output_states += [psi_out]

        p_list = []
        for in_state in input_states:
            for out_state in output_states:
                prod = inner_product(in_state, out_state)
                p_list += [prod]
        # |001>
        exp_list = [1.0, 0.0, 0.0, 0.0]
        # |010>
        exp_list += [0.0, 1.0, 0.0, 0.0]
        # |101>
        exp_list += [0.0, 0.0, np.cos(theta/2), complex(0, -np.sin(theta/2))]
        # |110> 
        exp_list += [0.0, 0.0, complex(0, -np.sin(theta/2)), np.cos(theta/2)]
        
        for result, expected in zip(p_list, exp_list):
            self.assertAlmostEqual(result, expected, places=6)
            # print(result, expected)

    def test_iterative_phase_estimation(self):
        theta = 5*np.pi/16
        # theta = np.pi/3 # -0.5235987755982988
        # print(-theta/2) 
        n_itter = 6
        n_qubits = 2
        a_idx = 1
        
        state = QuantumState(n_qubits) # |ancilla>|logical>
        kickback_phase = 0.0
        for k in reversed(range(1, n_itter)):
            psi = state.copy()
            phi = kickback_phase/2
            # print('k={}, phi={} mod (np.pi)'.format(k, phi))
            circuit = QuantumCircuit(n_qubits)
            # Apply H to ancilla bit to get |+> state
            circuit.add_H_gate(a_idx)
            # Apply kickback phase rotation to ancilla bit
            circuit.add_RZ_gate(a_idx, -np.pi * phi)
            # Apply C-U(Z0)
            theta_k = 2 ** (k-1) * theta
            # print('phase:{} mod (np.pi)'.format(theta_k/np.pi))
            circuit.add_RZ_gate(0, -theta_k)
            circuit.add_CNOT_gate(a_idx, 0)
            circuit.add_RZ_gate(0, theta_k)
            circuit.add_CNOT_gate(a_idx, 0)
            # Apply H to ancilla bit to get |+> state
            circuit.add_H_gate(a_idx)

            # run circuit
            circuit.update_quantum_state(psi)
            # print(psi.get_vector())
            # partial trace
            p0 = psi.get_marginal_probability([2, 0])
            p1 = psi.get_marginal_probability([2, 1])
            # print(p0, p1)
            # update kickback phase
            kth_digit = 1 if (p0 < p1) else 0
            kickback_phase = kickback_phase/2 + kth_digit
            # print(kickback_phase)
        # print(-np.pi * kickback_phase/2)
        self.assertAlmostEqual(-np.pi * kickback_phase/2, -theta/2)

    def suspend_test_phase_estimation_debug(self):
        theta = 5*np.pi/16
        n_qubits = 2
        a_idx = 1
        k = 2
        circuit = QuantumCircuit(n_qubits)
        psi = QuantumState(n_qubits) # |ancilla>|logical>
        phi = 1.25/2
        print('k={}, phi={} mod (np.pi)'.format(k, phi))
        # Apply H to ancilla bit to get |+> state
        circuit.add_H_gate(a_idx)
        # Apply kickback phase rotation to ancilla bit
        circuit.add_RZ_gate(a_idx, -np.pi * phi)
        # Apply C-U(Z0)
        theta_k = 2 ** (k-1) * theta
        print('phase:{} mod (np.pi)'.format(theta_k/np.pi))
        circuit.add_RZ_gate(0, -theta_k)
        circuit.add_CNOT_gate(a_idx, 0)
        circuit.add_RZ_gate(0, theta_k)
        circuit.add_CNOT_gate(a_idx, 0)
        # Apply H to ancilla bit to get |+> state
        circuit.add_H_gate(a_idx)

        # run circuit
        circuit.update_quantum_state(psi)
        print(psi.get_vector())

        # partial trace
        p0 = psi.get_marginal_probability([2, 0])
        p1 = psi.get_marginal_probability([2, 1])
        print(p0, p1)

if __name__ == "__main__":
    unittest.main()