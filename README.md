# Simple Addition Neural Network in PyTorch

A basic, beginner-friendly PyTorch project that trains a simple feedforward neural network to add two numbers together. This project serves as a practical introduction to building custom neural networks, defining loss functions, and using backpropagation.

## 🧠 Model Architecture
The network is built using PyTorch's `nn.Module` and consists of:
- **Input Layer:** 2 neurons (takes a pair of numbers).
- **Hidden Layer:** 8 neurons with a ReLU activation function.
- **Output Layer:** 1 neuron (outputs the predicted sum).

## ⚙️ Technologies Used
- **PyTorch:** For tensor operations and building the neural network.
- **Loss Function:** Mean Squared Error (`nn.MSELoss`).
- **Optimizer:** Stochastic Gradient Descent (`SGD`) with a learning rate of 0.001.

