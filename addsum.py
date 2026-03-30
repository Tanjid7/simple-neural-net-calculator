import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn

# Data: input pairs and their sums
x = [[1,2],[3,4],[5,6],[7,8]]
y = [[3],[7],[11],[15]]


# initialize using torch 
X = torch.tensor(x).float()
Y = torch.tensor(y).float()

# Define the model
class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer = nn.Linear(2, 8)  # 2 inputs → 8 neurons
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Linear(8, 1)  # 8 neurons → 1 output


    def forward(self, x):
        x = self.input_to_hidden_layer(x)
        x = self.hidden_layer_activation(x)
        x = self.hidden_to_output_layer(x)
        return x
    

# Create model, loss, and optimizer
mynet = MyNeuralNet()
loss_func = nn.MSELoss()
from torch.optim import SGD
opt = SGD(mynet.parameters(), lr=0.001)


# Training loop
for _ in range(2000):
    opt.zero_grad()           # 1. clear old gradients
    loss_value = loss_func(mynet(X), Y)  # 2. forward + compute loss
    loss_value.backward()     # 3. backpropagation
    opt.step()                # 4. update weights



# 1. Print the final loss to see how well it minimized the error
print(f"Final Loss: {loss_value.item():.4f}\n")

# 2. Get the model's predictions for the input data
predictions = mynet(X)

# 3. Print a comparison of the expected output vs actual prediction
print("Inputs   | Target Sum | Model Prediction")
print("-" * 45)

for i in range(len(X)):
    input_vals = x[i]              # The original Python list input (e.g., [1, 2])
    target = y[i][0]               # The actual sum (e.g., 3)
    pred = predictions[i][0].item() # The model's guess, converted back to a standard float
    
    print(f"{input_vals}   |      {target}     | {pred:.4f}")