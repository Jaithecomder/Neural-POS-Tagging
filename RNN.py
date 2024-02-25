import torch.nn as nn
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

def trainRNN(trainX, trainY, devX, devY, batchSize=32, hSize=128, numLayers=2, direction=2, epochs=20, lr=1e-3, device='cpu'):
    newTrainX = []
    newTrainY = []
    newDevX = []
    newDevY = []
    for i in range(len(trainX)):
        newTrainX.append(torch.tensor(trainX[i], dtype=torch.float).to(device))
        newTrainY.append(torch.tensor(trainY[i], dtype=torch.float).to(device))
    for i in range(len(devX)):
        newDevX.append(torch.tensor(devX[i], dtype=torch.float).to(device))
        newDevY.append(torch.tensor(devY[i], dtype=torch.float).to(device))

    trainDL = DataLoader(list(zip(newTrainX, newTrainY)), batch_size=batchSize, shuffle=True, collate_fn=lambda x: collate_fn(x, device))

    model = RNN(len(newTrainX[0][0]), len(newTrainY[0][0]), hSize, numLayers, direction, device=device).to(device)

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
            for i in range(len(newDevX)):
                output = model(newDevX[i])
                for j in range(len(devY[i])):
                    if output[j].argmax() == newDevY[i][j].argmax():
                        correct += 1
                    total += 1
            print('Dev Accuracy:', correct / total)
    return model


def testRNN(model, testX, testY, device='cpu'):
    newTestX = []
    newTestY = []
    for i in range(len(testX)):
        newTestX.append(torch.tensor(testX[i], dtype=torch.float).to(device))
        newTestY.append(torch.tensor(testY[i], dtype=torch.float).to(device))

    with torch.no_grad():
        correct = 0
        total = 0
        for i in range(len(newTestX)):
            output = model(newTestX[i])
            for j in range(len(testY[i])):
                if output[j].argmax() == newTestY[i][j].argmax():
                    correct += 1
                total += 1
    return correct / total
