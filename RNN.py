import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    
def trainRNN(trainX, trainY, devX, devY, device='cpu'):
    trainX = torch.tensor(trainX, dtype=torch.float).to(device)
    trainY = torch.tensor(trainY, dtype=torch.long).to(device)
    devX = torch.tensor(devX, dtype=torch.float).to(device)
    devY = torch.tensor(devY, dtype=torch.long).to(device)

    trainDL = DataLoader(list(zip(trainX, trainY)), batch_size=32, shuffle=True)

    model = RNN(len(trainX[0][0]), len(trainY[0]), 128).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(20):
        losses = []
        for i, data in enumerate(trainDL, 0):
            inputs, labels = data
            optimizer.zero_grad()
            hidden = model.initHidden()
            for i in range(len(inputs)):
                output, hidden = model(inputs[i], hidden)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch + 1} Loss: {sum(losses)/len(losses)}")
    return model