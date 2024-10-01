# useful additional packages
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

# optimization/optimize.py
def process_map_items(map_items):
    """
    Example function that processes map_items array.
    Here, we just return the number of map items for simplicity.
    """
    # print("Processing map items:", map_items)
    
    distance_matrix = np.round(10000*get_tsp_matrix(map_items),0)

    print("The distance matrix: ", distance_matrix)
    # Example: Count the number of items and return it
    
    n = len(map_items)
    G = nx.from_numpy_array(distance_matrix)

    tsp = Tsp(G)
    qp = tsp.to_quadratic_program()
    # print(qp.prettyprint())

    qp2qubo = QuadraticProgramToQubo()
    qubo = qp2qubo.convert(qp)
    qubitOp, offset = qubo.to_ising()
    print("Offset:", offset)
    print("Ising Hamiltonian:")
    # print(str(qubitOp))
    optimizer = SPSA(maxiter=300)
    ry = TwoLocal(qubitOp.num_qubits, "ry", "cz", reps=5, entanglement="linear")
    vqe = SamplingVQE(sampler=Sampler(), ansatz=ry, optimizer=optimizer)

    # result = vqe.compute_minimum_eigenvalue(qubitOp)
    vqe_optimizer = MinimumEigenOptimizer(vqe)

    # solve quadratic program
    result = vqe_optimizer.solve(qp)
    z = tsp.interpret(result)
    # print("Result is: ", result, z)
    # print("energy:", result.eigenvalue.real)
    # print("time:", result.optimizer_time)
    # x = tsp.sample_most_likely(result.eigenstate)
    # print("feasible:", qubo.is_feasible(x))
    
    # z = tsp.interpret(x)
    print("solution:", z)
    return z

def create_nodes_array(N, seed=None):
    """
    Creates array of random points of size N.
    """
    if seed:
        print("seed", seed)
        np.random.seed(seed)

    nodes_list = []
    for i in range(N):
        nodes_list.append(np.random.rand(2) * 10)
    return np.array(nodes_list)


def get_tsp_matrix(nodes_array):
    """
    Creates distance matrix out of given coordinates.
    """
    number_of_nodes = len(nodes_array)
    matrix = np.zeros((number_of_nodes, number_of_nodes))
    for i in range(number_of_nodes):
        for j in range(i, number_of_nodes):
            matrix[i][j] = distance_between_points(nodes_array[i], nodes_array[j])
            matrix[j][i] = matrix[i][j]
    return matrix


def distance_between_points(point_A, point_B):
    return np.sqrt((point_A[0] - point_B[0])**2 + (point_A[1] - point_B[1])**2)