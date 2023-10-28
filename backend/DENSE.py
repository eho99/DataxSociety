import torch
import torch.nn as nn
from typing import List

class VariableLayerFeedForwardNN(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int], activations: List[str]):
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

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)

        for i in range(len(self.activations)):
            x = self.hidden_layers[i](x)
            x = self.activations[i](x)
        
        x = self.output_layer(x)
        return x

    def train(input_size: int, output_size: int,
                hidden_sizes: List[int], activations: List[str],
                loss_function: str, lr: float, num_epochs: int,
                all_inputs: List[List[float]], all_outputs: List[List[float]]):
        
        assert len(all_inputs) == len(all_outputs)

        all_inputs = torch.tensor(all_inputs)
        all_outputs = torch.tensor(all_outputs)

        if all_outputs.dim() == 1:
            all_outputs.view(-1, 1)   
        
        assert all_outputs.shape[1] == output_size
        assert all_inputs.shape[1] == input_size

        for i in range(num_epochs):
            for j in range(all_inputs.shape[0]):
                input = all_inputs[j]
                output = all_outputs[j]

            


