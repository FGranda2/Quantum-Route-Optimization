from qiskit_alice_bob_provider import AliceBobLocalProvider
from qiskit import QuantumCircuit, execute, transpile
from qiskit.primitives import BackendSampler, BackendEstimator
from qiskit.quantum_info import SparsePauliOp

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import EfficientSU2
from qiskit.transpiler.layout import TranspileLayout

from typing import Union, List
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.optimize import minimize


provider = AliceBobLocalProvider()
# print(provider.backends())
backend = provider.get_backend('EMU:40Q:LOGICAL_TARGET')

def apply_layout(
    operator: SparsePauliOp,
    layout: TranspileLayout | List[int] | None,
    num_qubits: int | None = None
) -> SparsePauliOp:
    """Apply a transpiler layout to a SparsePauliOp

    Args:
        operator: The SparsePauliOp to apply the layout to.
        layout: Either a TranspileLayout, a list of integers or None.
                If both layout and num_qubits are none, a copy of the operator is
                returned.
        num_qubits: The number of qubits to expand the operator to. If not
                provided then if `layout` is a TranspileLayout the
                number of the transpiler output circuit qubits will be used by
                default. If `layout` is a list of integers the permutation
                specified will be applied without any expansion. If layout is
                None, the operator will be expanded to the given number of qubits.

    Returns:
        A new SparsePauliOp with the provided layout applied
    """
    if layout is None and num_qubits is None:
        return operator.copy()

    n_qubits = operator.num_qubits
    if isinstance(layout, TranspileLayout):
        n_qubits = len(layout._output_qubit_list)
        layout = layout.final_index_layout()
    if num_qubits is not None:
        if num_qubits < n_qubits:
            print("Error")
        n_qubits = num_qubits
    if layout is None:
        layout = list(range(operator.num_qubits))
    else:
        if any(x < 0 or x >= n_qubits for x in layout):
            print("Provided layout contains indices outside the number of qubits.")
        if len(set(layout)) != len(layout):
            print("Provided layout contains duplicate indices.")
    if operator.num_qubits == 0:
        return SparsePauliOp(["I" * n_qubits] * operator.size, operator.coeffs)
    new_op = SparsePauliOp("I" * n_qubits)
    return new_op.compose(operator, qargs=layout)

def interpret_tsp_result(x: Union[List[float], np.ndarray]) -> List[int]:
    """
    Interpret a TSP result as a list of node indices.

    Args:
        x (Union[List[float], np.ndarray]): The optimal x values representing the TSP solution.

    Returns:
        List[int]: A list of nodes representing the order of the TSP tour.

    Raises:
        ValueError: If the input cannot be interpreted as a valid TSP solution.
    """
    if isinstance(x, list):
        x = np.array(x)
    
    if not isinstance(x, np.ndarray):
        raise ValueError("Input must be a list or numpy array")

    n = int(np.sqrt(len(x)))
    if n * n != len(x):
        raise ValueError("Input length must be a perfect square")

    route = []
    for p in range(n):
        for i in range(n):
            if x[i * n + p] > 0.5:  # Use threshold for floating-point values
                route.append(i)
                break  # Assume only one city per position
    
    # if len(route) != n:
    #     raise ValueError("Invalid TSP solution: not all cities are visited")

    return route

def build_max_cut_paulis(Q: np.ndarray) -> list[tuple[str, float]]:
    """Convert the QUBO matrix Q to Pauli list for MaxCut."""
    n = Q.shape[0]
    pauli_list = []

    # Handle linear terms (diagonal elements)
    for i in range(n):
        if Q[i, i] != 0:
            paulis = ["I"] * n
            paulis[i] = "Z"
            pauli_list.append(("".join(paulis)[::-1], -Q[i, i]/2))

    # Handle quadratic terms (off-diagonal elements)
    for i in range(n):
        for j in range(i+1, n):
            if Q[i, j] != 0:
                paulis = ["I"] * n
                paulis[i], paulis[j] = "Z", "Z"
                pauli_list.append(("".join(paulis)[::-1],  Q[i, j]/4))

    return pauli_list

def compute_Q_with_constraints(distance_matrix, penalty):
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
            row_weights = 0 
            
            # Add the QUBO vector term (row elements)
            for j in range(n):
                row_weights += distance_matrix[i,j]
            Q[qubit, qubit] += 2*(row_weights*2)/4
            

        for p in range(n):
            qubit = i * n + p
            
            # Quadratic terms for row constraints
            for p2 in range(p+1, n):
                qubit2 = i * n + p2
                Q[qubit, qubit2] += 2 * penalty
            
            # Quadratic terms for column constraints
            for i2 in range(i+1, n):
                qubit2 = i2 * n + p
                Q[qubit, qubit2] += 2 * penalty

    return Q

def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]

# Number of nodes
n_bits = 3
#         [0, 400, 600, 800],
#         [400, 0, 300, 500],
#         [600, 300, 0, 700],
#         [800, 500, 700, 0]
# Create adjacency matrix
adj_matrix = np.array([[ 0, 48, 91,],[48,  0, 63,], [91, 63,  0,]])
# adj_matrix = np.array([[0, 400, 600, 800],[400, 0, 300, 500],[600, 300, 0, 700],[800, 500, 700, 0]])
# Create graph
G = nx.from_numpy_array(adj_matrix)

penalty = 1200
my_obj = compute_Q_with_constraints(adj_matrix, penalty)

# Do conversion to ising hamiltonian using custon function
pauli_result = build_max_cut_paulis(my_obj)
cost_hamiltonian = SparsePauliOp.from_list(pauli_result)

# Select target [TSP or custom]
target_program = cost_hamiltonian # Choose Custom

ansatz = EfficientSU2(target_program.num_qubits)
ansatz.measure_all()
pm = generate_preset_pass_manager(optimization_level=3)

ansatz_isa = pm.run(ansatz)
# hamiltonian_isa = target_program.apply_layout(layout=ansatz_isa.layout)
hamiltonian_isa = apply_layout(target_program, ansatz_isa.layout)
cost_history_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],
}

def cost_func(params, ansatz, hamiltonian, estimator):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (BackendEstimator): Estimator primitive instance

    Returns:
        float: Energy estimate
    """
    # Bind the parameters to the ansatz
    bound_circuit = ansatz.bind_parameters(params)

    # Run the estimator
    job = estimator.run(circuits=[bound_circuit], observables=[hamiltonian])
    result = job.result()
    energy = result.values[0]

    # Update cost history
    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(energy)
    print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {energy}]")
    return energy

num_params = ansatz.num_parameters
print("NUMBER OF PARAMETERS: ", num_params)
x0 = 2 * np.pi * np.random.random(num_params)
estimator = BackendEstimator(backend)

res = minimize(
        cost_func,
        x0,
        args=(ansatz_isa, hamiltonian_isa, estimator),
        method="COBYLA",
        tol=1e-2,
        options={'maxiter': 1000}  # Set maximum iterations to 5000
    )

print(res)

optimized_circuit = ansatz_isa.assign_parameters(res.x)

# Run the job
sampler = BackendSampler(backend)
shots = 10000
job = sampler.run([optimized_circuit], shots=shots)
result = job.result()

# Get the counts
counts = result.quasi_dists[0]

# Convert quasi-distribution to integer and binary counts
counts_int = counts
counts_bin = {k: v for k, v in counts.items()}
shots = sum(counts_int.values())
final_distribution_int = {key: val/shots for key, val in counts_int.items()}
final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}

# Process the results
keys = list(final_distribution_int.keys())
values = list(final_distribution_int.values())
most_likely = keys[np.argmax(np.abs(values))]
most_likely_bitstring = to_bitstring(most_likely, n_bits**2)
most_likely_bitstring.reverse()

print("Result bitstring:", most_likely_bitstring)
print("Path", interpret_tsp_result(most_likely_bitstring))