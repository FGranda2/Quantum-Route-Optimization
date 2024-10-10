import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Assuming you already have your adj_matrix defined
adj_matrix = np.array([[0, 400, 600, 800],[400, 0, 300, 500],[600, 300, 0, 1000],[800, 500, 1000, 0]])

# Create graph
G = nx.from_numpy_array(adj_matrix)

# Create a new figure
plt.figure(figsize=(8, 6))

# Draw the graph
pos = nx.spring_layout(G)  # Position nodes using spring layout
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=16, font_weight='bold')

# Draw edge labels
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Add a title
plt.title("TSP Graph Visualization", fontsize=16)

# Show the plot
plt.axis('off')  # Turn off the axis
plt.tight_layout()
plt.show()

from itertools import permutations


def brute_force_tsp(w, N):
    a = list(permutations(range(1, N)))
    last_best_distance = 1e10
    for i in a:
        distance = 0
        pre_j = 0
        for j in i:
            distance = distance + w[j, pre_j]
            pre_j = j
        distance = distance + w[pre_j, 0]
        order = (0,) + i
        if distance < last_best_distance:
            best_order = order
            last_best_distance = distance
            print("order = " + str(order) + " Distance = " + str(distance))
    return last_best_distance, best_order


best_distance, best_order = brute_force_tsp(adj_matrix, 4)
print(
    "Best order from brute force = "
    + str(best_order)
    + " with total distance = "
    + str(best_distance)
)