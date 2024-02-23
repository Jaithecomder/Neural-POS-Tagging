import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch

class ANN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
def dataPrep(trainX, trainY, contextSize, pad):
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
    return X, Y
    
def trainANN(trainX , trainY, devX, devY, pad, contextSize=2, device='cpu'):
    trainX, trainY = dataPrep(trainX, trainY, contextSize, pad)
    devX, devY = dataPrep(devX, devY, contextSize, pad)

    trainX = torch.tensor(trainX, dtype=torch.float)
    trainX = torch.flatten(trainX, start_dim=1).to(device)
    trainY = torch.tensor(trainY, dtype=torch.float).to(device)
    devX = torch.tensor(devX, dtype=torch.float)
    devX = torch.flatten(devX, start_dim=1).to(device)
    devY = torch.tensor(devY, dtype=torch.float).to(device)

    trainDL = DataLoader(list(zip(trainX, trainY)), batch_size=32, shuffle=True)

    model = ANN(len(trainX[0]), len(trainY[0]), 200).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(20):
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
            loss = criterion(output, devY.argmax(dim=1))
            print('Dev Loss:', loss.item())
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
        print('Test Accuracy:', correct / len(output))