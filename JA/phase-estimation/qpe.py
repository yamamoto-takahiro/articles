from functools import reduce
import numpy as np
from numpy.linalg import matrix_power, eig
from scipy.sparse.linalg import eigsh
from openfermion.ops import QubitOperator
from openfermion.transforms import get_sparse_operator
from qulacs import QuantumState, Observable, QuantumCircuit

"""
Implementation of IQPE algorithm for solving GE of H2 molecule
Ref:
Scalable Quantum Simulation of Molecular Energies,
PHYSICAL REVIEW X 6, 031007 (2016)
"""

def all_term_hamiltonian():
    """
    distance = 0.70 A
    """
    n_qubits = 2
    # H = -0.4584 I + 0.3593 Z0 − 0.4826 Z1 + 0.5818 Z0 Z1 + 0.0896 X0 X1 + 0.0896 Y0 Y1
    g_list = [-0.4584, 0.5818, 0.3593, 0.0896, -0.4826, 0.0896]
    pauli_strings = ['', 'Z0 Z1', 'Z0', 'Y0 Y1', 'Z1', 'X0 X1']
    hamiltonian = QubitOperator()
    for g, h in zip(g_list, pauli_strings):
        hamiltonian += g * QubitOperator(h)
    # {{0.0001,0,0,0}, {0,-0.1983,0.1792,0}, {0,0.1792,-1.8821,0}, {0,0,0,0.2467}}
    sparse_matrix = get_sparse_operator(hamiltonian, n_qubits=n_qubits)
    vals, vecs = eigsh(sparse_matrix, k=1, which='SA') # -1.90096027
    return sparse_matrix, vals

def reduced_term_hamiltonian():
    """
    distance = 0.70 A
    removed 'I' and 'Z0 Z1' terms, which add up to -1.31916027
    """
    n_qubits = 2
    g_list = [0.3593, 0.0896, -0.4826, 0.0896]
    pauli_strings = ['Z0', 'Y0 Y1', 'Z1', 'X0 X1']
    hamiltonian = QubitOperator()
    for g, h in zip(g_list, pauli_strings):
        hamiltonian += g * QubitOperator(h)
    sparse_matrix = get_sparse_operator(hamiltonian, n_qubits=n_qubits)
    vals, vecs = eigsh(sparse_matrix, k=1, which='SA') # -0.86076027
    return sparse_matrix, vals

def diag_term_hamiltonian():
    """
    distance = 0.70 A
    contains only: 'Z0', 'Z1' terms, which add up to 
    """
    n_qubits = 2
    g_list = [0.3593, -0.4826]
    pauli_strings = ['Z0', 'Z1']
    hamiltonian = QubitOperator()
    for g, h in zip(g_list, pauli_strings):
        hamiltonian += g * QubitOperator(h)
    sparse_matrix = get_sparse_operator(hamiltonian, n_qubits=n_qubits)
    vals, vecs = eigsh(sparse_matrix, k=1, which='SA')
    return sparse_matrix, vals

def reduced_hamiltonian_HWE():
    """
    Ref:
    Hardware-efficient Variational Quantum Eigensolver for Small Molecules and Quantum Magnets,
    https://arxiv.org/pdf/1704.05018.pdf
    """
    g_list = [0.011280, 0.397936, 0.397936, 0.180931]
    pauli_strings = ['Z0 Z1', 'Z0', 'Z1', 'X0 X1']
    hamiltonian = QubitOperator()
    for g, h in zip(g_list, pauli_strings):
        hamiltonian += g * QubitOperator(h)
    sparse_matrix = get_sparse_operator(hamiltonian, n_qubits=n_qubits)
    vals, vecs = eigsh(sparse_matrix, k=1, which='SA') # -0.80489907
    return sparse_matrix, vals

def order_n_trotter_suzuki_approx(t, n_trotter_steps):
    """
    ordering: 'Z0', 'Y0 Y1', 'Z1', 'X0 X1'
    Returns:
        sparse_matrix: trotterized [exp(iHt/n)]^n
        args: list of phases of each eigenvalue exp(i*phase)
    """
    n_qubits = 2
    g_list = [0.3593, 0.0896, -0.4826, 0.0896]
    pauli_strings = ['Z0', 'Y0 Y1', 'Z1', 'X0 X1']
    terms = []
    for g, h in zip(g_list, pauli_strings):
        arg = g * t / n_trotter_steps
        qop = complex(np.cos(arg), 0) * QubitOperator('') - complex(0, np.sin(arg)) * QubitOperator(h)
        terms += [get_sparse_operator(qop, n_qubits=n_qubits)]
    sparse_matrix = reduce(np.dot, terms)
    matrix = matrix_power(sparse_matrix.toarray(), n_trotter_steps)
    vals, vecs = eig(matrix)
    args = np.angle(vals)
    return sparse_matrix, sorted(args)
    # {0.99643289+0.07688573I,0,0,0.03387043-0.00792882I}
    # {0,0.85222297-0.51169358I, -0.02484937-0.1061518I,0}
    # {0,0.02484937-0.1061518I,0.85222297+0.51169358I,0}
    # {-0.03387043-0.00792882I,0,0,0.99643289-0.07688573I}

    # Wolfram format (reduced)
    # {0.996433+0.076886I,0,0,0.03387-0.007929I}
    # {0,0.852223-0.511694I, -0.02485-0.10615I,0}
    # {0,0.024845-0.10615I,0.852223+0.511694I,0}
    # {-0.03387-0.007929I,0,0,0.996433-0.076886I}
    # {{0.996433+0.076886I,0,0,0.03387-0.007929I},{0,0.852223-0.511694I, -0.02485-0.10615I,0},{0,0.024845-0.10615I,0.852223+0.511694I,0},{-0.03387-0.007929I,0,0,0.996433-0.076886I}}
    # {0.852224 + 0.523179 I, 0.996433 + 0.084389 I, 0.996433 - 0.084389 I, 0.852222 - 0.523179 I}
    # phase_deg = {31.5457, 4.84089, -4.84089, -31.5457}
    # phase_rad = {0.5505763298, 0.08448946923, -0.08448946923, -0.5505763298}
    # E = -0.5505763298/0.640 = -0.8602755153125 (t=0.640, n_trotter=1)

def time_evolution_operator(n_qubits, a_idx, t, hamiltonian, n_trotter_step):
    circuit = QuantumCircuit(n_qubits)
    n_terms = hamiltonian.get_term_count()
    for _ in range(n_trotter_step):
        for i in range(n_terms):
            pauli_term = hamiltonian.get_term(i)
            c = pauli_term.get_coef()
            pauli_ids = pauli_term.get_pauli_id_list()
            pauli_indices = pauli_term.get_index_list()
            param = - c * t / n_trotter_step
            circuit.add_multi_Pauli_rotation_gate(pauli_indices, pauli_ids, angle)
    return circuit

def ctrl_RZ_circuit(theta_k, kickback_phase):
    n_qubits = 2
    a_idx = 1
    phi = kickback_phase/2
    circuit = QuantumCircuit(n_qubits)
    # Apply H to ancilla bit to get |+> state
    circuit.add_H_gate(a_idx)
    # Apply kickback phase rotation to ancilla bit
    circuit.add_RZ_gate(a_idx, -np.pi * phi)
    # Apply C-U(Z0)
    # print('phase:{} mod (np.pi)'.format(theta_k/np.pi))
    circuit.add_RZ_gate(0, -theta_k)
    circuit.add_CNOT_gate(a_idx, 0)
    circuit.add_RZ_gate(0, theta_k)
    circuit.add_CNOT_gate(a_idx, 0)
    # Apply H to ancilla bit to get |+> state
    circuit.add_H_gate(a_idx)
    return circuit

def time_evolution_circuit(g_list, t, kickback_phase, n_trotter_step=1):
    n_qubits = 3
    a_idx = 2
    phi = -(t / n_trotter_step) * g_list
    # print(phi)
    circuit = QuantumCircuit(n_qubits)
    circuit.add_H_gate(a_idx)
    # Apply kickback phase rotation to ancilla bit
    circuit.add_RZ_gate(a_idx, -np.pi*kickback_phase/2)
    for _ in range(n_trotter_step):
        # CU(Z0)
        circuit.add_RZ_gate(0, -phi[0])
        circuit.add_CNOT_gate(a_idx, 0)
        circuit.add_RZ_gate(0, phi[0])
        circuit.add_CNOT_gate(a_idx, 0)
        
        # CU(Y0 Y1)
        circuit.add_S_gate(0)
        circuit.add_S_gate(1)
        circuit.add_H_gate(0)
        circuit.add_H_gate(1)
        circuit.add_CNOT_gate(1, 0)
        circuit.add_RZ_gate(0, -phi[1])
        circuit.add_CNOT_gate(a_idx, 0)
        circuit.add_RZ_gate(0, phi[1])
        circuit.add_CNOT_gate(a_idx, 0)
        circuit.add_CNOT_gate(1, 0)                
        circuit.add_H_gate(0)
        circuit.add_H_gate(1)
        circuit.add_Sdag_gate(0)
        circuit.add_Sdag_gate(1)
        
        # CU(Z1)
        circuit.add_RZ_gate(1, -phi[2])
        circuit.add_CNOT_gate(a_idx, 1)
        circuit.add_RZ_gate(1, phi[2])
        circuit.add_CNOT_gate(a_idx, 1)
        
        # CU(X0 X1)
        circuit.add_H_gate(0)
        circuit.add_H_gate(1)
        circuit.add_CNOT_gate(1, 0)
        circuit.add_RZ_gate(0, -phi[3])
        circuit.add_CNOT_gate(a_idx, 0)
        circuit.add_RZ_gate(0, phi[3])
        circuit.add_CNOT_gate(a_idx, 0)
        circuit.add_CNOT_gate(1, 0)     
        circuit.add_H_gate(0)
        circuit.add_H_gate(1)
        
    circuit.add_H_gate(a_idx)
    return circuit

def time_evolution_circuit_improved(g_list, t, kickback_phase, k, n_trotter_step=1):
    n_qubits = 3
    a_idx = 2
    phi = -(t / n_trotter_step) * g_list
    # print(phi)
    circuit = QuantumCircuit(n_qubits)
    circuit.add_H_gate(a_idx)
    # Apply kickback phase rotation to ancilla bit
    circuit.add_RZ_gate(a_idx, -np.pi*kickback_phase/2)
    for _ in range(n_trotter_step):
        for _ in range(2 ** k):
            # CU(Z0)
            circuit.add_RZ_gate(0, -phi[0])
            circuit.add_CNOT_gate(a_idx, 0)
            circuit.add_RZ_gate(0, phi[0])
            circuit.add_CNOT_gate(a_idx, 0)
        
            # CU(Y0 Y1)
            circuit.add_S_gate(0)
            circuit.add_S_gate(1)
            circuit.add_H_gate(0)
            circuit.add_H_gate(1)
            circuit.add_CNOT_gate(1, 0)
            circuit.add_RZ_gate(0, -phi[1])
            circuit.add_CNOT_gate(a_idx, 0)
            circuit.add_RZ_gate(0, phi[1])
            circuit.add_CNOT_gate(a_idx, 0)
            circuit.add_CNOT_gate(1, 0)                
            circuit.add_H_gate(0)
            circuit.add_H_gate(1)
            circuit.add_Sdag_gate(0)
            circuit.add_Sdag_gate(1)
        
            # CU(Z1)
            circuit.add_RZ_gate(1, -phi[2])
            circuit.add_CNOT_gate(a_idx, 1)
            circuit.add_RZ_gate(1, phi[2])
            circuit.add_CNOT_gate(a_idx, 1)
        
            # CU(X0 X1)
            circuit.add_H_gate(0)
            circuit.add_H_gate(1)
            circuit.add_CNOT_gate(1, 0)
            circuit.add_RZ_gate(0, -phi[3])
            circuit.add_CNOT_gate(a_idx, 0)
            circuit.add_RZ_gate(0, phi[3])
            circuit.add_CNOT_gate(a_idx, 0)
            circuit.add_CNOT_gate(1, 0)     
            circuit.add_H_gate(0)
            circuit.add_H_gate(1)
        
    circuit.add_H_gate(a_idx)
    return circuit

def iterative_phase_estimation(g_list, t, n_itter, init_state, n_trotter_step=1, kickback_phase=0.0):
    for k in reversed(range(1, n_itter)):
        psi = init_state.copy()
        phi = kickback_phase/2
        # g_k_list = 2 ** (k-1) * np.array(g_list)
        # circuit = time_evolution_circuit(g_k_list, t, kickback_phase, n_trotter_step=n_trotter_step)
        g_k_list = np.array(g_list)
        circuit = time_evolution_circuit_improved(g_k_list, t, kickback_phase, k, n_trotter_step=n_trotter_step)
        circuit.update_quantum_state(psi)
        # partial trace
        p0 = psi.get_marginal_probability([2, 2, 0])
        p1 = psi.get_marginal_probability([2, 2, 1])
        # update kickback phase
        kth_digit = 1 if (p0 < p1) else 0
        kickback_phase = kickback_phase/2 + kth_digit
    return -0.5 * np.pi * kickback_phase
    
def iterative_phase_estimation_general(n_qubits, a_idx, circuit, theta, n_itter, init_state, kickback_phase=0.0):  
    for k in reversed(range(1, n_itter)):
        psi = init_state.copy()
        phi = kickback_phase/2
        # TODO: update parameter of parametric circuit
        circuit.update_quantum_state(psi)
        # partial trace
        p0 = psi.get_marginal_probability([lambda i : 0 if i == a_idx else 2 for i in range(n_qubits)])
        p1 = psi.get_marginal_probability([lambda i : 1 if i == a_idx else 2 for i in range(n_qubits)])
        # update kickback phase
        kth_digit = 1 if (p0 < p1) else 0
        kickback_phase = kickback_phase/2 + kth_digit
    return -0.5 * np.pi * kickback_phase

if __name__ == "__main__":
    n_qubits = 3 # 2 for electron configurations and 1 for ancilla
    g_list = [0.3593, 0.0896, -0.4826, 0.0896]
    pauli_strings = ['Z 0', 'Y 0 Y 1', 'Z 1', 'X 0 X 1']
    hf_state = QuantumState(n_qubits)
    hf_state.set_computational_basis(0b001) # |0>|01>
    t = 0.640
    n_itter = 16 # determine precission

    # validity check    
    _, eigs = reduced_term_hamiltonian()
    e_exact = eigs[0]
    print('e_exact:{}'.format(e_exact))
    print('n, e_matrix, e_iqpe, |e_matrix-e_iqpe|, |e_exact-e_iqpe|')
    for n in range(1, 10, 2):
        iqpe_phase = iterative_phase_estimation(g_list, t, n_itter, hf_state, n_trotter_step=n, kickback_phase=0.0)
        e_iqpe = iqpe_phase/t
        _, phases = order_n_trotter_suzuki_approx(t, n)
        e = phases[0]/t
        print(n, e, e_iqpe, abs(e-e_iqpe), abs(e_exact-e_iqpe))
    """
    Output:
    n, e_matrix, e_iqpe, |e_matrix-e_iqpe|, |e_exact-e_iqpe|
    1 -0.8602760325707504 -0.6693273070834999 0.1909487254872505 0.19143296732511783
    3 -0.860706856078986 -0.7079012829293991 0.15280557314958687 0.15285899147921855
    5 -0.8607410547561056 -0.919124264750132 0.05838320999402635 0.058363990341514294
    7 -0.8607504699997903 -0.7912669789002896 0.0694834910995007 0.06949329550832806
    9 -0.8607543437287754 -0.8634068690596158 0.002652525330840483 0.0026465946509981464
    11 -0.8607563044136105 -0.7810041334607948 0.07975217095281573 0.0797561409478229
    13 -0.8607574320394854 -0.8526873841004028 0.008070047939082614 0.008072890308214897
    15 -0.8607581394987095 -0.8851180109222472 0.024359871423537682 0.02435773651362949
    17 -0.8607586122946013 -0.8452389993805676 0.015519612914033698 0.015521275028050119
    19 -0.8607589438039048 -0.8616445667646808 0.0008856229607759891 0.0008842923560631322
    21 -0.860759185187819 -0.8476015475142109 0.013157637673608114 0.01315872689440678
    23 -0.8607593663841868 -0.89463236939779 0.03387300301360319 0.03387209498917232
    25 -0.8607595058585225 -0.8420535569632025 0.018705948895319957 0.01870671744541519
    27 -0.8607596155019882 -0.857658583882912 0.0031010316190761555 0.003101690525705636
    29 -0.8607597032525749 -0.8490904331538153 0.011669270098759621 0.011669841254802416
    31 -0.8607597745733818 -0.8614591377532399 0.0006993631798580813 0.0006988633446222137
    33 -0.8607598333239603 -0.8360584962547744 0.02470133706918598 0.024701778153843335
    35 -0.860759882293679 -0.8418863676965448 0.018873514597134156 0.018873906712072896
    37 -0.8607599235390256 -0.8503252673276852 0.010434656211340432 0.010435007080932479
    39 -0.8607599586031012 -0.8379780333165446 0.022781925286556537 0.022782241092073052
    41 -0.8607599886620682 -0.849034047821082 0.011725940840986104 0.011726226587535638
    43 -0.8607600146250893 -0.8658993751448198 0.005139360519730518 0.005139100736202162
    45 -0.8607600372038958 -0.8539224225518159 0.006837614652079882 0.0068378518568017466
    47 -0.8607600569620794 -0.8661181976063638 0.005358140644284415 0.005357923197746084
    49 -0.8607600743506145 -0.86967832268983 0.008918248339215507 0.008918048281212343
    """