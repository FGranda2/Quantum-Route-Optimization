import numpy as np

# optimization/optimize.py
def process_map_items(map_items):
    """
    Example function that processes map_items array.
    Here, we just return the number of map items for simplicity.
    """
    print("Processing map items:", map_items)
    
    distance_matrix = get_tsp_matrix(map_items)

    print("The distance matrix: ", distance_matrix)
    # Example: Count the number of items and return it
    return len(map_items)

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