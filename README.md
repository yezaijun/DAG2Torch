# Custom Graph-Based Neural Network Framework

This project provides a framework for building and visualizing custom neural network architectures based on a Directed Acyclic Graph (DAG). It includes a Python script for defining, analyzing, and executing custom neural network topologies using PyTorch.

Modify the `CustomGraphNetwork.create_layers` method to adapt the script to more PyTorch components. This enhancement is flexible and easy to implement.


## Features

- Custom Graph Definition: The framework uses a user-defined DAG to represent the architecture of the neural network.
- Dynamic Layer Creation: Supports automatic layer creation based on the provided graph configuration.
- Graph Pruning: Automatically removes unused or unreachable nodes from the graph.
- Visualization: Generates a visual representation of the final pruned DAG.
- Topological Execution: Ensures the network executes in the correct order by using topological sorting.

## How It Works

1.	Define the Network as a DAG:
	- The architecture of the neural network is represented as a graph with nodes as layers and edges defining the flow of data.
	- Input the DAG’s edges, start node, and end node via a JSON configuration file.
2.	Prune and Analyze the Graph:
	- The script ensures only nodes that are reachable from the start node and can reach the end node are included in the graph.
	- Performs topological sorting to determine execution order.
3.	Build Neural Network Layers:
	- Each node in the DAG corresponds to a specific type of PyTorch layer, such as Conv2d, MaxPool2d, BatchNorm2d, etc.
	- The script dynamically creates these layers based on user-specified configurations.
4.	Execute the Network:
	- Perform a forward pass on the network using the topological order of the DAG.

## Requirements
- Python 3.8+
- PyTorch 1.8+
- NetworkX (for DAG visualization)
- Matplotlib (for DAG visualization)
- Graphviz (for DAG visualization)


## License

This project is open-sourced under the MIT License. See the LICENSE file for details.