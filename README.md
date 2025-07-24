# Particle Identification with Graph Neural Network
* This project implements a Graph Neural Network (GNN) for particle identification from physics event data, using PyTorch for neural network operations
and networkx plus matplotlib for visualization. It builds graph representations dynamically by combining electron momentum data with jet and photon multiplicity. 
The simple graph convolution layers manually normalize adjacency matrices and apply learned linear transformations.
The Streamlit interface enables interactive exploration: users can browse multiple particle datasets, customize GNN architecture and training parameters, 
start training on CPU/GPU, monitor training and validation losses/accuracies per epoch, and visualize example event graphs.
Core functionalities include:

# Data Loading and Preprocessing:
Reads multiple physics event datasets, constructs node features, and defines graph adjacency matrices.
# Manual Graph Convolution Layer:
Implements core GCN layer with adjacency normalization and ReLU activation.
# Multi-layer GNN Model:
Stacks several graph convolution layers with dropout and global mean pooling for classification.
# Training and Evaluation:
Includes standard training loops with cross-entropy loss and accuracy calculation.
# Visualization:
Converts adjacency and node feature tensors to Network for plot rendering within Streamlit.
# User Control:
Adjustable hidden layer size, number of layers, dropout rate, epochs, and learning rate, allowing real-time experimentation.
# Device Selection: 
Supports training on GPU if available.
