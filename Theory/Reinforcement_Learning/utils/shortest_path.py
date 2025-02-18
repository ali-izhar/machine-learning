from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt


class ShortestPathDP:
    def __init__(self, graph=None):
        """
        Initialize ShortestPathDP with an optional graph.

        Args:
            graph (dict or list): Either a dict of dicts representing adjacency list,
                                or a list of tuples (from_node, to_node, weight)
        """
        # Initialize graph as adjacency list
        self.graph = defaultdict(dict)
        self.distances = {}  # Store shortest distances
        self.predecessors = {}  # Store predecessors for path reconstruction

        # If graph is provided, initialize it
        if graph is not None:
            self.load_graph(graph)

    def load_graph(self, graph):
        """
        Load a graph from either an adjacency list or edge list.

        Args:
            graph: Either a dict of dicts (adjacency list) or
                  list of tuples (from_node, to_node, weight)
        """
        if isinstance(graph, dict):
            # Load from adjacency list
            for from_node, edges in graph.items():
                for to_node, weight in edges.items():
                    self.add_edge(from_node, to_node, weight)
        elif isinstance(graph, list):
            # Load from edge list
            for from_node, to_node, weight in graph:
                self.add_edge(from_node, to_node, weight)
        else:
            raise ValueError("Graph must be either a dict or list of edges")

    def add_edge(self, from_node, to_node, weight):
        """Add an edge to the graph."""
        self.graph[from_node][to_node] = weight

    def get_topological_order(self):
        """
        Get topological order of nodes using Kahn's algorithm.
        Returns list of nodes in topological order.
        """
        # Create a copy of the graph
        in_degree = defaultdict(int)
        for node in self.graph:
            for neighbor in self.graph[node]:
                in_degree[neighbor] += 1

        # Add all nodes with no incoming edges to queue
        queue = [node for node in self.graph if in_degree[node] == 0]
        topo_order = []

        while queue:
            node = queue.pop(0)
            topo_order.append(node)

            # Reduce in-degree of neighbors
            for neighbor in self.graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return topo_order

    def solve(self, start, end):
        """
        Solve the shortest path problem using dynamic programming.

        Args:
            start: Starting node
            end: End node

        Returns:
            tuple: (shortest_distance, path)
        """
        if not self.graph:
            raise ValueError("Graph is empty. Load a graph first.")

        # Initialize distances
        nodes = set(self.graph.keys()).union(
            {node for edges in self.graph.values() for node in edges}
        )
        self.distances = {node: float("inf") for node in nodes}
        self.distances[start] = 0
        self.predecessors = {node: None for node in nodes}

        # Get topological order
        topo_order = self.get_topological_order()
        if not topo_order:
            raise ValueError("Graph contains a cycle. Must be a DAG.")

        # Add end node if not in topological order
        if end not in topo_order:
            topo_order.append(end)

        # Dynamic Programming step
        print("\nDynamic Programming Steps:")
        print("Initial distances:", self.distances)

        for node in topo_order:
            if node in self.graph:
                current_dist = self.distances[node]
                # Relax all edges going from current node
                for neighbor, weight in self.graph[node].items():
                    new_dist = current_dist + weight
                    if new_dist < self.distances[neighbor]:
                        self.distances[neighbor] = new_dist
                        self.predecessors[neighbor] = node
                print(f"After processing {node}:", self.distances)

        # Reconstruct path
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = self.predecessors[current]
        path.reverse()

        return self.distances[end], path

    def visualize(self, path=None):
        """Visualize the graph and highlight the shortest path if provided."""
        G = nx.DiGraph()

        # Add edges to NetworkX graph
        for from_node, edges in self.graph.items():
            for to_node, weight in edges.items():
                G.add_edge(from_node, to_node, weight=weight)

        # Set up the plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500)
        nx.draw_networkx_labels(G, pos)

        # Draw edges with weights
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels)

        # Draw regular edges
        nx.draw_networkx_edges(G, pos)

        # Highlight the shortest path if provided
        if path:
            path_edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="r", width=2)

        plt.title("Shortest Path Visualization")
        plt.axis("off")
        plt.show()


def main():
    # Example graph as edge list
    example_graph = [
        ("S", "A", 20),
        ("S", "B", 10),
        ("A", "C", 10),
        ("B", "C", 5),
        ("B", "D", 15),
        ("C", "D", 10),
        ("C", "E", 20),
        ("D", "E", 5),
        ("D", "F", 15),
        ("E", "T", 10),
        ("F", "T", 20),
    ]

    # Create solver with the example graph
    solver = ShortestPathDP(example_graph)

    print("Finding shortest path from S to T...")
    distance, path = solver.solve("S", "T")

    print("\nResults:")
    print(f"Shortest distance: {distance} miles")
    print(f"Shortest path: {' -> '.join(path)}")

    # Visualize the solution
    solver.visualize(path)


if __name__ == "__main__":
    main()
