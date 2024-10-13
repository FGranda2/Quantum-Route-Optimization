# useful additional packages
from typing import Union, List
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit_algorithms import SamplingVQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
import networkx as nx

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from scipy.optimize import minimize
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
# from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit_ibm_runtime import EstimatorV2, SamplerV2

from qiskit_aer import AerSimulator

aer_sim = AerSimulator()

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

def add_constraints_to_Q(Q, n, penalty):
    # Constraint for each city (row constraints)
    for i in range(n):
        for p1 in range(n):
            qubit1 = i * n + p1
            Q[qubit1, qubit1] += penalty  # Add penalty to diagonal
            for p2 in range(p1+1, n):
                qubit2 = i * n + p2
                Q[qubit1, qubit2] += 2 * penalty  # Off-diagonal terms

    # Constraint for each position (column constraints)
    for p in range(n):
        for i1 in range(n):
            qubit1 = i1 * n + p
            Q[qubit1, qubit1] += penalty  # Add penalty to diagonal
            for i2 in range(i1+1, n):
                qubit2 = i2 * n + p
                Q[qubit1, qubit2] += 2 * penalty  # Off-diagonal terms

    # Subtract the linear terms
    for i in range(n):
        for p in range(n):
            qubit = i * n + p
            Q[qubit, qubit] -= 2 * penalty

    return Q

def compute_Q_for_tsp_with_binary_variables(distance_matrix):
    n = distance_matrix.shape[0]  # Number of cities
    Q = np.zeros((n**2, n**2))  # Initialize the QUBO matrix (n^2 x n^2)

    # Iterate over each position in the tour
    for p in range(n):  # Current position in the tour
        for i in range(n):  # Current city
            for j in range(n):  # Next city
                if i == j:
                    continue  # Skip self-loops

                # Calculate qubit indices
                qubit_a = i * n + p
                qubit_b = j * n + ((p + 1) % n)

                # Ensure we only fill the upper triangular part
                if qubit_a < qubit_b:
                    Q[qubit_a, qubit_b] += distance_matrix[i,j]  # Add weight from city i to j
                elif qubit_a > qubit_b:
                    Q[qubit_b, qubit_a] += distance_matrix[i,j]  # Add weight from city i to j
                # When qubit_a == qubit_b, we don't add anything (diagonal elements)

     # Add constraint terms
    Q = add_constraints_to_Q(Q, n, 91)

    return Q

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
            row_weights = 0 
            
            # Add the QUBO vector term (diagonal elements)
            for j in range(n):
                row_weights += distance_matrix[i,j]
            Q[qubit, qubit] += 2*(row_weights*(n-1))/4
            

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

n_bits = 4
# [[ 0. 48. 91.]
#  [48.  0. 63.]
#  [91. 63.  0.]]
adj_matrix = np.array([[ 0, 48, 91,],[48,  0, 63,], [91, 63,  0,]])
adj_matrix = np.array([[0, 400, 600, 800],[400, 0, 300, 500],[600, 300, 0, 700],[800, 500, 700, 0]])
G = nx.from_numpy_array(adj_matrix)

tsp = Tsp(G)
qp = tsp.to_quadratic_program()
print("QP PROBLEM", qp.prettyprint())
print("--------------------------------------------------------------------------")
print("OBJECTIVE:")
print(qp.objective.quadratic.to_array())
print("--------------------------------------------------------------------------")
penalty = 1200
my_obj = compute_Q_for_tsp_with_constraints_v2(adj_matrix, penalty)
# my_obj = compute_Q_for_tsp_with_binary_variables(adj_matrix)
print("MY:")
print(my_obj)


qp2qubo = QuadraticProgramToQubo(penalty)
qubo = qp2qubo.convert(qp)

# Get the number of variables
sense = qubo.objective.sense.value
# print("QUBO PENALTY: ", qubo.objective.)
print("SENSE: ", sense)
print("QUBO: ", qubo)
print("LINEAR TERMS: ", qubo.objective.linear.to_dict().items())
print("QUBO Constant TERM:", qubo.objective.constant)
print("QUADRATIC TERMS: ", qubo.objective.quadratic)
for idx, coef in qubo.objective.linear.to_dict().items():
    print("---",idx,coef)

qubitOp, offset = qubo.to_ising()
print("Offset:", offset)
print("Ising Hamiltonian:")
print(str(qubitOp.coeffs))

print("--------------------------------------------------------------------------")
pauli_result = build_max_cut_paulis(my_obj)
cost_hamiltonian = SparsePauliOp.from_list(pauli_result)
print(cost_hamiltonian.coeffs)
repetitions = 1
circuit = QAOAAnsatz(cost_operator=qubitOp, reps=repetitions)
circuit.measure_all()

circuit.draw('mpl')
# plt.show()

pm = generate_preset_pass_manager(optimization_level=3)

candidate_circuit = pm.run(circuit)

initial_gamma = 1.6
initial_beta = 2.9
init_params = [val for _ in range(repetitions) for val in [initial_gamma, initial_beta]]
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
estimator = EstimatorV2(mode=aer_sim)

result = minimize(
    cost_func_estimator,
    init_params,
    args=(candidate_circuit, qubitOp, estimator),
    method="COBYQA",
    tol=1e-7,
)
print(result)

# plt.figure(figsize=(12, 6))
# plt.plot(objective_func_vals)
# plt.xlabel("Iteration")
# plt.ylabel("Cost")
# plt.show()

optimized_circuit = candidate_circuit.assign_parameters(result.x)

sampler = SamplerV2(mode=aer_sim)
shots = 500000

pub= (optimized_circuit, )
job = sampler.run([pub], shots=shots)
counts_int = job.result()[0].data.meas.get_int_counts()
counts_bin = job.result()[0].data.meas.get_counts()
shots = sum(counts_int.values())
final_distribution_int = {key: val/shots for key, val in counts_int.items()}
final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}
# print(final_distribution_int)

def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]

keys = list(final_distribution_int.keys())
values = list(final_distribution_int.values())
most_likely = keys[np.argmax(np.abs(values))]
print("Most Likely: ", most_likely)
most_likely_bitstring = to_bitstring(most_likely, n_bits**2)
most_likely_bitstring.reverse()

print("Result bitstring:", most_likely_bitstring)
print("Path", interpret_tsp_result(most_likely_bitstring))