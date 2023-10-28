import torch
import torch.nn as nn
import torch.optim as optim

class LSTM(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int, dropout: float):
        
        super(LSTM, self).__init()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.linear(out)
        return out

    def train_data(self, train_data, train_labels, epochs, learning_rate, binary):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        if binary:
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            outputs = self(train_data)
            loss = criterion(outputs, train_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        print('Training completed.')
    
    def evaluate_model(self, test_data, test_labels, binary):

        # come back to
        self.eval()

        with torch.no_grad():
            outputs = self(test_data)

            threshold = 0.5
            predicted_labels = (outputs > threshold).float()

            correct = (predicted_labels == test_labels).sum().item()
            total = len(test_labels)
            accuracy = (correct / total) * 100

            print(f'Accuracy: {accuracy:.2f}%')
            return accuracy
        
