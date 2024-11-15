from collections import defaultdict, deque
import torch
import torch.nn as nn
import json
import copy
# for graph visualization
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
# for graph visualization

class DAG:
    def __init__(self, edges, start_node, end_node):
        self.graph = defaultdict(list)
        self.indegree = defaultdict(int)
        self.all_nodes = set()
        self.start_node, self.end_node = start_node, end_node

        self.build_graph(edges)
        pruned_nodes = self.prune_graph()
        if pruned_nodes:
            print("The following nodes were not properly connected and have been skipped:", pruned_nodes)

        self.topo_sort_result = self._topological_sort()
        if self.topo_sort_result:
            print("Topological sorting:", self.topo_sort_result)
        else:
            print("The graph contains a cycle and cannot be topologically sorted.")

        self.reverse_graph = self._reverse_graph()

    def prune_graph(self):
        """Remove nodes that cannot be reached from the start or cannot reach the end, returning the pruned nodes."""
        reachable_from_start = self._reachable_nodes(self.start_node, forward=True)
        reachable_to_end = self._reachable_nodes(self.end_node, forward=False)

        valid_nodes = reachable_from_start & reachable_to_end
        pruned_nodes = self.all_nodes - valid_nodes

        # Remove unused nodes
        for node in pruned_nodes:
            self.graph.pop(node, None)
            self.indegree.pop(node, None)
            for neighbors in self.graph.values():
                if node in neighbors:
                    neighbors.remove(node)

        self.all_nodes = valid_nodes  # Update the valid nodes set
        return pruned_nodes

    def _reachable_nodes(self, start, forward=True):
        """Return the set of all nodes reachable from the start or end node."""
        reachable = set()
        queue = deque([start])
        graph = self.graph if forward else self._reverse_graph()

        while queue:
            node = queue.popleft()
            if node not in reachable:
                reachable.add(node)
                queue.extend(graph[node])
        
        return reachable

    def _reverse_graph(self):
        """Generate the reverse of the current graph."""
        reverse_graph = defaultdict(list)
        for node, neighbors in self.graph.items():
            for neighbor in neighbors:
                reverse_graph[neighbor].append(node)
        return reverse_graph

    def visualize_graph(self, figsize=(10, 8)):
        """Visualize the final pruned graph."""
        G = nx.DiGraph()
        for node, neighbors in self.graph.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        
        pos = graphviz_layout(G, prog="dot")  # Use graphviz_layout for layout
        plt.figure(figsize=figsize)
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue",
                font_size=15, font_weight="bold", arrowstyle="->", arrowsize=15)
        plt.title("Layout of Directed Acyclic Graph (DAG)")
        plt.show()

    def _topological_sort(self):
        """Return a list of nodes in topological order. If the graph contains a cycle, return an empty list."""
        queue = deque([node for node in self.indegree if self.indegree[node] == 0])
        topo_order = []

        while queue:
            node = queue.popleft()
            topo_order.append(node)
            for neighbor in self.graph[node]:
                self.indegree[neighbor] -= 1
                if self.indegree[neighbor] == 0:
                    queue.append(neighbor)
        
        return topo_order if len(topo_order) == len(self.all_nodes) else []

    def _add_edge(self, start, end):
        """Add a directed edge and update indegree and node sets."""
        self.graph[start].append(end)
        self.indegree[end] += 1
        self.all_nodes.update([start, end])
        if start not in self.indegree:
            self.indegree[start] = 0

    def build_graph(self, edges):
        """Build the graph from a list of edges."""
        for start, end in edges:
            self._add_edge(start, end)

        # Ensure start and end nodes are part of the graph
        self.all_nodes.update([self.start_node, self.end_node])
        self.indegree.setdefault(self.start_node, 0)
        self.indegree.setdefault(self.end_node, 0)


class CustomGraphNetwork(nn.Module):
    def __init__(self, DAG: DAG, node_config, input_shape):
        super(CustomGraphNetwork, self).__init__()
        self.DAG = DAG
        self.node_config = node_config
        self.input_shape = input_shape

        self.layers = nn.ModuleDict()  # Store layers/modules for the graph
        self.layers_output_shape = {}  # Store the output shapes of each node

        # Step 1: Build layers
        self.create_layers()

        # Step 2: Define forward execution order based on topological sorting
        self.execution_order = self.DAG.topo_sort_result

    def create_layers(self):
        """Create each layer/module based on the node configuration."""
        for node, config in self.node_config.items():
            self.create_single_layer(node)

    def create_single_layer(self, node):
        """Create a single layer/module based on its type and configuration."""
        config = copy.deepcopy(self.node_config[node])
        layer_type = config.pop('type')

        # Determine the input shape from parent nodes
        father_nodes = self.DAG.reverse_graph[node]

        if len(father_nodes) >= 2 and layer_type != 'add':
            raise ValueError(f"Error: Multiple input nodes for node {node} with type {layer_type}.")
        elif len(father_nodes) == 0:
            input_shape = self.input_shape  # Start node
        elif len(father_nodes) == 1:
            input_shape = self.get_layer_output_shape(father_nodes[0])  # Single input node
        elif layer_type == 'add':
            input_shapes = [self.get_layer_output_shape(father) for father in father_nodes]
            if not all(s == input_shapes[0] for s in input_shapes):
                raise ValueError(f"Error: Inconsistent input channels for node {node}.")
            input_shape = input_shapes[0]
        else:
            raise ValueError(f"Unknown layer type {layer_type} for node {node}.")

        # Layer creation logic
        if layer_type == 'Conv2d':
            config['in_channels'] = input_shape[0]
            self.layers[node] = getattr(nn, layer_type)(**config)
            self.layers_output_shape[node] = self._calculate_output_shape(input_shape, config)
        elif layer_type in ['MaxPool2d', 'AvgPool2d']:
            self.layers[node] = getattr(nn, layer_type)(**config)
            self.layers_output_shape[node] = self._calculate_output_shape(input_shape, config)
        elif layer_type == 'BatchNorm2d':
            config['num_features'] = input_shape[0]
            self.layers[node] = getattr(nn, layer_type)(**config)
            self.layers_output_shape[node] = input_shape
        elif layer_type == 'activation':
            self.layers[node] = getattr(nn, config['fun'])()
            self.layers_output_shape[node] = input_shape
        elif layer_type == 'add':
            self.layers[node] = None  # No operation for add
            self.layers_output_shape[node] = input_shape
        elif layer_type == 'Linear':
            config['in_features'] = input_shape[0] * input_shape[1] * input_shape[2]
            self.layers[node] = nn.Sequential(
                nn.Flatten(start_dim=1),
                getattr(nn, layer_type)(**config)
            )
            self.layers_output_shape[node] = (config['out_features'], 0, 0)
        else:
            raise ValueError(f"Unknown layer type {layer_type} for node {node}.")

    def get_layer_output_shape(self, node):
        """Retrieve the output shape of a given node."""
        output_shape = self.layers_output_shape.get(node, None)
        if output_shape is None:
            raise ValueError(f"Input node {node} for node {node} is not defined.")
        return output_shape

    def _calculate_output_shape(self, input_shape, config):
        """Calculate the output shape of a layer."""
        kernel_size = config['kernel_size']
        stride = config.get('stride', 1)
        padding = config.get('padding', 0)
        compute_dim = lambda x: (x + 2 * padding - kernel_size) // stride + 1
        if config.get('out_channels', None):
            output_shape = (config['out_channels'], compute_dim(input_shape[1]), compute_dim(input_shape[2]))
        else:
            output_shape = (input_shape[0], compute_dim(input_shape[1]), compute_dim(input_shape[2]))
        return output_shape

    def forward(self, x):
        """Define the forward pass based on topological order."""
        node_outputs = {self.DAG.start_node: self.layers[self.DAG.start_node](x)}

        for node in self.execution_order[1:]:
            node_operator = self.layers[node]
            parent_nodes = self.DAG.reverse_graph[node]

            if node_operator:  # Regular layer
                node_outputs[node] = node_operator(node_outputs[parent_nodes[0]])
            else:  # Add layer
                inputs = [node_outputs[parent] for parent in parent_nodes]
                node_outputs[node] = sum(inputs)

        return node_outputs[self.DAG.end_node]


if __name__ == "__main__":
    # Load JSON configuration
    file_path = 'test_CNN.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    myCNNgraph = DAG(edges=data['edges'], start_node=data['start_node'], end_node=data['end_node'])
    myCNNgraph.visualize_graph(figsize=(5, 5))

    myCNNmodel = CustomGraphNetwork(myCNNgraph, data['nodes'], input_shape=(1, 8, 8))

    # Generate random binary images
    batch_size, height, width, channels = 10, 8, 8, 1
    binary_images = torch.randint(0, 2, (batch_size, channels, height, width)).float()
    print("Binary Images Shape:", binary_images.shape)

    output = myCNNmodel(binary_images)
    print("Output Shape:", output.shape)