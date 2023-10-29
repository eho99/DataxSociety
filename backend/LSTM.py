import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from typing import List
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int, dropout: float):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers  # Define num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.linear(out)
        return out

    def train_model(self, train_loader, epochs, learning_rate, loss_function):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

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

        for epoch in range(epochs):
            self.train()
            for batch_data, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = self(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        print('Training completed.')

    def evaluate_model(self, test_loader):
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                outputs = self(batch_data)
                _, predicted = torch.max(outputs, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        accuracy = (correct / total) * 100
        print(f'Accuracy: {accuracy:.2f}%')
        return accuracy
'''''
Testing
input_size = 10
output_size = 3
hidden_size = 64
num_layers = 2
dropout = 0.2
num_samples = 100
seq_length = 18
batch_size = 18  # Match the batch size

# Generate random data and labels with the correct batch size
data = torch.rand((num_samples, seq_length, input_size))
labels = torch.randint(0, output_size, (num_samples,))

# Split the data into a training and testing set
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)

# Create data loaders with the matching batch size
train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=batch_size)
lstm_model = LSTM(input_size, output_size, hidden_size, num_layers, dropout)

# Train the model
lstm_model.train_model(train_loader, epochs=50, learning_rate=0.001, loss_function="CrossEntropyLoss")

# Evaluate the model
lstm_model.evaluate_model(test_loader)
'''





