import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from typing import List

# ASSUME FOR A CLASSICATION PROBLEM 

class VariableLayerFeedForwardNN(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int], activations: List[str], possible_labels: List[str]):
        assert len(hidden_sizes) == len(activations)

        super(VariableLayerFeedForwardNN, self).__init__()
        
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.activations = []
        
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

        for activation_function in activations:
            if activation_function == "relu":
                self.activations.append(nn.ReLU())
            elif activation_function == "sigmoid":
                self.activations.append(nn.Sigmoid())
            elif activation_function == "tanh":
                self.activations.append(nn.Tanh())
            else:
                raise ValueError("Unsupported activation function in list!")
            
        self.possible_labels = possible_labels

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)

        for i in range(len(self.activations)):
            x = self.hidden_layers[i](x)
            x = self.activations[i](x)
        
        x = self.output_layer(x)
        return x

    def _train(self, loss_function: str, lr: float, num_epochs: int, 
              train_inputs: List[List[float]], train_outputs: List[List[float]]):
        # uses Adam optimizer
        assert len(train_inputs) == len(train_outputs)

        train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
        train_outputs = torch.tensor(train_outputs, dtype=torch.float32)

        if train_outputs.dim() == 1:
            train_outputs = train_outputs.view(-1, 1)

        # Find the loss function based on the input string
        if loss_function == "L1Loss":
            criterion = nn.L1Loss()
        elif loss_function == "MSELoss":
            criterion = nn.MSELoss()
        elif loss_function == "NLLLoss":
            criterion = nn.NLLLoss()
        elif loss_function == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("Unsupported loss function!")
        
        optimizer = optim.Adam(self.parameters(), lr=lr)

        epoch_status = []

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            outputs = self(train_inputs)
            loss = criterion(outputs, train_outputs)

            loss.backward()
            optimizer.step()

            running_loss = loss.item()
            epoch_status.append(f'Epoch {epoch + 1}, Loss: {running_loss}')
        return epoch_status 
        
    def eval(self, test_inputs, test_outputs):
        outputs = self(test_inputs)
        best_indices = torch.argmax(outputs)








            


