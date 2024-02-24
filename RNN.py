import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

class RNN(nn.Module):
    def __init__(self, inputSize, outputSize, hiddenSize, numLayers=1, activation='tanh'):
        super(RNN, self).__init__()
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.rnn = nn.RNN(inputSize, hiddenSize, batch_first=True, num_layers=numLayers, nonlinearity=activation)
        self.fc1 = nn.Linear(hiddenSize, 128)
        self.fc2 = nn.Linear(128, outputSize)

    def forward(self, input):
        output, hidden = self.rnn(input)
        output = self.fc1(output)
        output = torch.relu(output)
        output = self.fc2(output)
        output = torch.softmax(output, dim=1)
        return output, hidden
    
def collate_fn(batch):
    data = [x[0] for x in batch]
    target = [x[1] for x in batch]
    data = pad_sequence(data, batch_first=True)
    target = pad_sequence(target, batch_first=True)
    return [data, target]
    
def trainRNN(trainX, trainY, devX, devY, device='cpu'):
    for i in range(len(trainX)):
        trainX[i] = torch.tensor(trainX[i], dtype=torch.float).to(device)
        trainY[i] = torch.tensor(trainY[i], dtype=torch.float).to(device)
    for i in range(len(devX)):
        devX[i] = torch.tensor(devX[i], dtype=torch.float).to(device)
        devY[i] = torch.tensor(devY[i], dtype=torch.float).to(device)

    # trainX = pad_sequence(trainX, batch_first=True)
    # trainY = pad_sequence(trainY, batch_first=True)
    # devX = pad_sequence(devX, batch_first=True)
    # devY = pad_sequence(devY, batch_first=True)
    batchSize = 32
    trainDL = DataLoader(list(zip(trainX, trainY)), batchSize=batchSize, shuffle=True, collate_fn=collate_fn)

    hSize = 128
    numLayers = 1

    model = RNN(len(trainX[0][0]), len(trainY[0][0]), hSize, numLayers).to(device)

    epochs = 10
    lr = 6e-4
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        losses = []
        for i, data in enumerate(trainDL, 0):
            inputs, labels = data
            optimizer.zero_grad()
            output, hidden = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print('----------------------------------')
        print('Epoch:', epoch, 'Loss:', sum(losses) / len(losses))
        with torch.no_grad():
            # hidden = model.initHidden(len(devX))
            # output, hidden = model(devX, hidden)
            # loss = criterion(output, devY)
            # print('Dev Loss:', loss.item())
            correct = 0
            total = 0
            for i in range(len(devX)):
                output, hidden = model(devX[i].unsqueeze(0))
                for j in range(len(devY[i])):
                    if output[0][j].argmax() == devY[i][j].argmax():
                        correct += 1
                    total += 1
            print('Dev Accuracy:', correct / total)
    return model

def testRNN(model, testX, testY, device='cpu'):
    with torch.no_grad():
        for i in range(len(testX)):
            testX[i] = torch.tensor(testX[i], dtype=torch.float).to(device)
            testY[i] = torch.tensor(testY[i], dtype=torch.float).to(device)
        testX = pad_sequence(testX, batch_first=True)
        output, hidden = model(testX)
        correct = 0
        total = 0
        for i in range(len(testY)):
            for j in range(len(testY[i])):
                if output[i][j].argmax() == testY[i][j].argmax():
                    correct += 1
                total += 1

        print('Test Accuracy:', correct / total)
        