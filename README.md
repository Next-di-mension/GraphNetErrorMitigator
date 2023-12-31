# Graph Net Based Error Mitigation

## Overview

This project seeks to contribute to quantum computing by employing Graph Neural Networks (GNNs) for error mitigation. The focus is on enhancing the precision and stability of quantum computations in environments with quantum noise, utilizing the unique capabilities of GNNs.

## Methodology

### Training Data Formulation
The foundation of our methodology is the formulation of training data using $T_2$ operators characterized by distinct parameter configurations. Essential features of the training data include:
- Graphical representations of excitation operators.
- Quantification of noisy expectation values.
- Enumeration of two-qubit and one-qubit gates.

### Graph Formulation
Our approach utilizes a weighted directed graph to model qubit interactions, emphasizing various $T_2$ operator configurations. This graph is bifurcated into two subsets, each representing a specific group of qubits. The graph's edges are weighted in accordance with the frequency of qubit exchanges.

### Neural Network Architecture
The project employs two neural network paradigms:
- A Graph Neural Network (GNN) dedicated to processing the graphical data representing qubit interactions.
- A Fully Connected Neural Network for the regression analysis of noisy expectation values and gate counts.

These networks' outputs are subsequently amalgamated and input into a simple feed-forward neural network for enhanced training.

## Objectives
The primary aim of this research is to refine quantum error mitigation strategies through the application of GNNs. By accurately modeling quantum system behaviors, this project intends to mitigate the effects of noise and errors in quantum computations, thereby contributing to the development of more robust and efficient quantum computing capabilities.

# Installation

# Citation