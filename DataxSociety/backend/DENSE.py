import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from typing import List

# ASSUME FOR A CLASSICATION PROBLEM 

class DENSE(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int], activations: List[str], possible_labels: List[str]):
        assert len(hidden_sizes) == len(activations)
        assert len(possible_labels) == output_size

        super(DENSE, self).__init__()
        
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.activations = []

        self.loss_func = None
        
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

        for activation_function in activations:
            if activation_function == "ReLU":
                self.activations.append(nn.ReLU())
            elif activation_function == "Sigmoid":
                self.activations.append(nn.Sigmoid())
            elif activation_function == "Tanh":
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
        x = torch.nn.functional.softmax(x, dim=0)
        return x

    def _train(self, loss_function: str, lr: float, num_epochs: int, 
              train_inputs: List[List[float]], train_outputs: List[List[float]]):
        num_epochs = int(num_epochs)
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
        
        self.loss_func = criterion
        
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
    
    def eval(self, inputs):
        outputs = self(inputs)
        return torch.argmax(outputs)
    
    def accuracy(self, inputs, output_labels):
        output_labels = torch.tensor(output_labels)
        model_outputs = self.eval(inputs)
        matches = torch.sum(model_outputs == output_labels)
        return matches / len(inputs)


    def _test_loss(self, test_inputs, test_outputs):
        # returns testing loss:
        outputs = self(test_inputs)
        loss = self.loss_func(outputs, test_outputs)

        return loss.item()        
        
    def train(self, loss_function: str, lr: float, num_epochs: int, test_ratio: float, 
              all_inputs: List[List[float]], all_output_labels: List[int]):

        encoder = torch.eye(len(self.possible_labels))
        all_outputs = []

        for label in all_output_labels:
            all_outputs.append(encoder[label].t())

        all_outputs = torch.stack(all_outputs, dim=0)
        all_inputs = torch.tensor(all_inputs)

        train_inputs, test_inputs, train_outputs, test_outputs, _, test_labels = train_test_split(all_inputs, all_outputs, all_output_labels, test_size=test_ratio)

        epoch_status = self._train(loss_function, lr, num_epochs, train_inputs, train_outputs)
        test_loss = self._test_loss(test_inputs, test_outputs)
        test_acuracy = self.accuracy(test_inputs, test_labels)

        return test_loss, epoch_status, test_acuracy
    

# network = DENSE(5, 10, [10, 10], ["ReLU", "ReLU"], [0,1,2,3,4,5,6,7,8,9])
# print(network.train("CrossEntropyLoss", 1e-4, 1e2, 0.5, [[1,2,3,4,5], [1,3,5,7,9]], [3, 5]))