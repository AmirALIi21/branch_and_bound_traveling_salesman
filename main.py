import numpy as np
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

class Node:
    def __init__(self, bound, level, path):
        self.bound = bound
        self.level = level
        self.path = path

class TSPBranchAndBound:
    def __init__(self, cost_matrix):
        self.N = len(cost_matrix)
        self.cost_matrix = np.array(cost_matrix)
        self.final_path = []
        self.final_cost = float('inf')
        self.tree_edges = []  # Store edges for visualization
        self.node_labels = {}  # Store node labels for visualization
        self.node_counter = 0

    def calculate_bound(self, curr_path):
        bound = 0
        
        # Create a copy of the cost matrix
        reduced_matrix = self.cost_matrix.copy()
        
        # Row reduction
        for i in range(self.N):
            if i not in curr_path:
                min_row = float('inf')
                for j in range(self.N):
                    if j not in curr_path and reduced_matrix[i][j] < min_row:
                        min_row = reduced_matrix[i][j]
                if min_row != float('inf'):
                    bound += min_row
                    reduced_matrix[i] -= min_row
        
        # Column reduction
        for j in range(self.N):
            if j not in curr_path:
                min_col = float('inf')
                for i in range(self.N):
                    if i not in curr_path and reduced_matrix[i][j] < min_col:
                        min_col = reduced_matrix[i][j]
                if min_col != float('inf'):
                    bound += min_col
                    reduced_matrix[:, j] -= min_col
        
        # Add cost of current path
        for i in range(len(curr_path) - 1):
            bound += self.cost_matrix[curr_path[i]][curr_path[i + 1]]
            
        return bound

    def solve(self):
        # Initialize with root node
        root_path = [0]  # Start from vertex 0
        root_bound = self.calculate_bound(root_path)
        root = Node(root_bound, 0, root_path)
        
        # Create priority queue (list in this case)
        pq = [root]
        self.node_labels[self.node_counter] = f"0\nb={root_bound}"
        self.node_counter += 1
        
        while pq:
            # Get node with minimum bound
            curr_node = min(pq, key=lambda x: x.bound)
            pq.remove(curr_node)
            
            # If we've reached a leaf node
            if len(curr_node.path) == self.N:
                # Add return to start
                total_cost = curr_node.bound + self.cost_matrix[curr_node.path[-1]][0]
                if total_cost < self.final_cost:
                    self.final_cost = total_cost
                    self.final_path = curr_node.path + [0]
                continue
            
            # Generate children
            parent_id = self.node_counter - 1
            for next_city in range(self.N):
                if next_city not in curr_node.path:
                    new_path = curr_node.path + [next_city]
                    new_bound = self.calculate_bound(new_path)
                    
                    # Only add node if its bound is less than current best solution
                    if new_bound < self.final_cost:
                        new_node = Node(new_bound, curr_node.level + 1, new_path)
                        pq.append(new_node)
                        
                        # Add edge and label for visualization
                        self.tree_edges.append((parent_id, self.node_counter))
                        self.node_labels[self.node_counter] = f"{next_city}\nb={new_bound:.1f}"
                        self.node_counter += 1

    def visualize_search_tree(self):
        G = nx.Graph()
        
        # Add nodes
        for node_id in self.node_labels:
            G.add_node(node_id)
        
        # Add edges
        G.add_edges_from(self.tree_edges)
        
        # Create the layout
        pos = nx.spring_layout(G)
        
        # Draw the graph
        plt.figure(figsize=(15, 10))
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, self.node_labels)
        
        plt.title("Branch and Bound Search Tree")
        plt.axis('off')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Cost matrix from the image
    cost_matrix = [
        [0, 14, 4, 10, 20,11],
        [14, 0, 7, 8, 7,9],
        [4, 5, 0, 7, 16,3],
        [11, 7, 9, 0, 2,6],
        [18, 7, 17, 4, 0,20]
    ]
    
    # Create TSP solver instance
    tsp = TSPBranchAndBound(cost_matrix)
    
    # Solve the problem
    tsp.solve()
    
    # Print results
    print("Optimal Path:", ' -> '.join(map(str, tsp.final_path)))
    print("Optimal Cost:", tsp.final_cost)
    
    # Visualize the search tree
    tsp.visualize_search_tree()