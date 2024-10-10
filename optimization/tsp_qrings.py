# useful additional packages
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from scipy.optimize import minimize
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit import qasm3

import QuantumRingsLib
from QuantumRingsLib import QuantumRegister, AncillaRegister, ClassicalRegister, QuantumCircuit
from QuantumRingsLib import QuantumRingsProvider
from QuantumRingsLib import job_monitor
from QuantumRingsLib import JobStatus

# Get provider
provider = QuantumRingsProvider(token ="rings-200.FhZKD8kgz5Un6hcTvEBb1LD6QppvVX5I", name="pancho.fg23@hotmail.com")
backend = provider.get_backend("scarlet_quantum_rings")
shots = 100

provider.active_account()


def build_max_cut_paulis(Q: np.ndarray) -> list[tuple[str, float]]:
    """Convert the QUBO matrix Q to Pauli list for MaxCut."""
    n = Q.shape[0]
    pauli_list = []

    # Handle linear terms (diagonal elements)
    for i in range(n):
        if Q[i, i] != 0:
            paulis = ["I"] * n
            paulis[i] = "Z"
            pauli_list.append(("".join(paulis)[::-1], -Q[i, i]))

    # Handle quadratic terms (off-diagonal elements)
    for i in range(n):
        for j in range(i+1, n):
            if Q[i, j] != 0:
                paulis = ["I"] * n
                paulis[i], paulis[j] = "Z", "Z"
                pauli_list.append(("".join(paulis)[::-1],  Q[i, j]/4))

    return pauli_list

def compute_Q_for_tsp_with_constraints_v2(distance_matrix, penalty):
    n = distance_matrix.shape[0]  # Number of cities
    Q = np.zeros((n**2, n**2))  # Initialize the QUBO matrix (n^2 x n^2)

    # Add distance terms (quadratic terms)
    for p in range(n):
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                qubit_a = i * n + p
                qubit_b = j * n + ((p + 1) % n)
                if qubit_a < qubit_b:
                    Q[qubit_a, qubit_b] += distance_matrix[i,j]
                elif qubit_a > qubit_b:
                    Q[qubit_b, qubit_a] += distance_matrix[i,j]

    # Add constraint terms
    for i in range(n):
        for p in range(n):
            qubit = i * n + p
            
            # Linear terms (diagonal elements)
            Q[qubit, qubit] += -2*penalty * (1 - 2)  # -penalty from (x - 1)^2 expansion
            
            # Quadratic terms for row constraints
            for p2 in range(p+1, n):
                qubit2 = i * n + p2
                Q[qubit, qubit2] += 2 * penalty
            
            # Quadratic terms for column constraints
            for i2 in range(i+1, n):
                qubit2 = i2 * n + p
                Q[qubit, qubit2] += 2 * penalty

    return Q

n_bits = 3
adj_matrix = np.array([[ 0, 48, 91,],[48,  0, 63,], [91, 63,  0,]])

G = nx.from_numpy_array(adj_matrix)

penalty = 1300
my_obj = compute_Q_for_tsp_with_constraints_v2(adj_matrix, penalty)

pauli_result = build_max_cut_paulis(my_obj)
cost_hamiltonian = SparsePauliOp.from_list(pauli_result)
print(cost_hamiltonian.coeffs)
repetitions = n_bits
circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=repetitions)
circuit.measure_all()

circuit.draw('mpl')
# plt.show()

pm = generate_preset_pass_manager(optimization_level=3)

candidate_circuit = pm.run(circuit)

initial_gamma = np.pi
initial_beta = np.pi*2
init_params = [initial_beta]*repetitions + [initial_gamma]*repetitions
print("INIT PARAMS: " , init_params)
objective_func_vals = [] # Global variable

def cost_func_estimator(params, ansatz, hamiltonian, estimator):

    # transform the observable defined on virtual qubits to
    # an observable defined on all physical qubits
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])

    results = job.result()[0]
    cost = results.data.evs

    objective_func_vals.append(cost)


    return cost

# If using qiskit-ibm-runtime<0.24.0, change `mode=` to `session=`
estimator = EstimatorV2()
result = minimize(
    cost_func_estimator,
    init_params,
    args=(candidate_circuit, cost_hamiltonian, estimator),
    method="COBYLA",
    tol=1e-2,
)
print(result)


optimized_circuit = candidate_circuit.assign_parameters(result.x)

qasm_str = qasm3.dumps(optimized_circuit)
sampler = SamplerV2()
shots = 1000

pub= (optimized_circuit, )
# job = sampler.run([pub], shots=shots)
qrings_circ = QuantumCircuit.from_qasm_str(qasm_str)
job = backend.run(qrings_circ, shots)
job_monitor(job)
counts_int = job.result()[0].data.meas.get_int_counts()
counts_bin = job.result()[0].data.meas.get_counts()
shots = sum(counts_int.values())
final_distribution_int = {key: val/shots for key, val in counts_int.items()}
final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}

def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]

keys = list(final_distribution_int.keys())
values = list(final_distribution_int.values())
most_likely = keys[np.argmax(np.abs(values))]
most_likely_bitstring = to_bitstring(most_likely, n_bits**2)
most_likely_bitstring.reverse()

print("Result bitstring:", most_likely_bitstring)