import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch

class ANN(nn.Module):
    def __init__(self, inputSize, outputSize, hiddenSizes, activation='relu', device='cpu'):
        super(ANN, self).__init__()
        self.fn = nn.ReLU()
        if activation == 'relu':
            self.fn = nn.ReLU()
        elif activation == 'tanh':
            self.fn = nn.Tanh()
        elif activation == 'sigmoid':
            self.fn = nn.Sigmoid()
        self.fc1 = nn.Linear(inputSize, hiddenSizes[0])
        self.hfcs = []
        for i in range(1, len(hiddenSizes)):
            self.hfcs.append(nn.Linear(hiddenSizes[i - 1], hiddenSizes[i]).to(device))
        self.fc2 = nn.Linear(hiddenSizes[-1], outputSize)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fn(x)
        for hfc in self.hfcs:
            x = hfc(x)
            x = self.fn(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=1)
    
def dataPrep(trainX, trainY, contextSize, pad, device='cpu'):
    X = []
    Y = []
    for s in range(len(trainX)):
        sentence = trainX[s]
        for i in range(len(sentence)):
            x = []
            y = []
            for j in range(i - contextSize, i + contextSize + 1):
                if j < 0 or j >= len(sentence):
                    x.append(pad)
                else:
                    x.append(sentence[j])
            X.append(x)
            Y.append(trainY[s][i])
    X = torch.tensor(X, dtype=torch.float)
    X = torch.flatten(X, start_dim=1).to(device)
    Y = torch.tensor(Y, dtype=torch.float).to(device)
    return X, Y
    
def trainANN(trainX , trainY, devX, devY, pad, contextSize=2, lr=0.001, hiddenSizes=(256,), activation='sigmoid', batchSize=32, epochs=20, device='cpu'):
    trainX, trainY = dataPrep(trainX, trainY, contextSize, pad, device)
    devX, devY = dataPrep(devX, devY, contextSize, pad, device)

    trainDL = DataLoader(list(zip(trainX, trainY)), batch_size=batchSize, shuffle=True)

    model = ANN(len(trainX[0]), len(trainY[0]), hiddenSizes, activation, device).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        losses = []
        for i, data in enumerate(trainDL, 0):
            inputs, labels = data
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels.argmax(dim=1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print('----------------------------------')
        print('Epoch:', epoch, 'Loss:', sum(losses) / len(losses))
        with torch.no_grad():
            output = model(devX)
            correct = 0
            for i in range(len(output)):
                if torch.argmax(output[i]) == torch.argmax(devY[i]):
                    correct += 1
            print('Dev Accuracy:', correct / len(output))
    return model

def testANN(model, testX, testY, pad, contextSize=2, device='cpu'):
    testX, testY = dataPrep(testX, testY, contextSize, pad)
    testX = torch.tensor(testX, dtype=torch.float)
    testX = torch.flatten(testX, start_dim=1).to(device)
    testY = torch.tensor(testY, dtype=torch.float).to(device)

    with torch.no_grad():
        output = model(testX)
        correct = 0
        for i in range(len(output)):
            if torch.argmax(output[i]) == torch.argmax(testY[i]):
                correct += 1
    return correct / len(output)