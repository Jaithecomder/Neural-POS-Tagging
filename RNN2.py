import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

class RNN(nn.Module):
    def __init__(self, inputSize, outputSize, hiddenSize, numLayers=1, direction=1, activation='tanh', device='cpu'):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.direction = direction
        self.device = device
        self.rnn = nn.RNN(inputSize, hiddenSize, batch_first=True, num_layers=numLayers, nonlinearity=activation, bidirectional=direction==2)
        self.fc1 = nn.Linear(direction * hiddenSize, outputSize)

    def forward(self, input):
        # hidden = torch.zeros(self.numLayers * self.direction, input.size(0), self.hiddenSize).to(self.device)
        output, hidden = self.rnn(input)
        output = self.fc1(output)
        output = torch.softmax(output, dim=1)
        return output
    
def collate_fn(batch, device='cpu'):
    data = [x[0] for x in batch]
    target = [x[1] for x in batch]
    data = pad_sequence(data, batch_first=True).to(device)
    target = pad_sequence(target, batch_first=True).to(device)
    return [data, target]

def trainRNN(trainX, trainY, devX, devY, device='cpu'):
    for i in range(len(trainX)):
        trainX[i] = torch.tensor(trainX[i], dtype=torch.float).to(device)
        trainY[i] = torch.tensor(trainY[i], dtype=torch.float).to(device)
    for i in range(len(devX)):
        devX[i] = torch.tensor(devX[i], dtype=torch.float).to(device)
        devY[i] = torch.tensor(devY[i], dtype=torch.float).to(device)

    batchSize = 32
    trainDL = DataLoader(list(zip(trainX, trainY)), batchSize=batchSize, shuffle=True, collate_fn=lambda x: collate_fn(x, device))

    hSize = 512
    numLayers = 1
    direction = 1

    model = RNN(len(trainX[0][0]), len(trainY[0][0]), hSize, numLayers, direction, device=device).to(device)

    epochs = 20
    lr = 1e-3
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        losses = []
        for i, (x, y) in enumerate(trainDL):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print('----------------------------------')
        print(f'Epoch {epoch+1}/{epochs} Loss: {sum(losses)/len(losses)}')
        with torch.no_grad():
            correct = 0
            total = 0
            for i in range(len(devX)):
                output = model(devX[i].unsqueeze(0))
                for j in range(len(devY[i])):
                    if output[0][j].argmax() == devY[i][j].argmax():
                        correct += 1
                    total += 1
            print('Dev Accuracy:', correct / total)
    return model


def testRNN(model, testX, testY, device='cpu'):
    for i in range(len(testX)):
        testX[i] = torch.tensor(testX[i], dtype=torch.float).to(device)
        testY[i] = torch.tensor(testY[i], dtype=torch.float).to(device)

    with torch.no_grad():
        correct = 0
        total = 0
        for i in range(len(testX)):
            output = model(testX[i].unsqueeze(0))
            for j in range(len(testY[i])):
                if output[0][j].argmax() == testY[i][j].argmax():
                    correct += 1
                total += 1
        print('Test Accuracy:', correct / total)
