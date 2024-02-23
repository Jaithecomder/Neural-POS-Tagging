import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, numLayers=1, activation='tanh'):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.numLayers = numLayers
        self.fc0 = nn.Linear(input_size, 256)
        self.rnn = nn.RNN(256, hidden_size, batch_first=True, num_layers=numLayers, nonlinearity=activation)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.fc(output)
        output = F.softmax(output, dim=1)
        return output, hidden
    
    def initHidden(self, batchSize):
        return torch.zeros(self.numLayers, batchSize, self.hidden_size)
    
def trainRNN(trainX, trainY, devX, devY, device='cpu'):
    for i in range(len(trainX)):
        trainX[i] = torch.tensor(trainX[i], dtype=torch.float).to(device)
        trainY[i] = torch.tensor(trainY[i], dtype=torch.float).to(device)
    for i in range(len(devX)):
        devX[i] = torch.tensor(devX[i], dtype=torch.float).to(device)
        devY[i] = torch.tensor(devY[i], dtype=torch.float).to(device)

    trainX = pad_sequence(trainX, batch_first=True)
    trainY = pad_sequence(trainY, batch_first=True)
    devX = pad_sequence(devX, batch_first=True)
    devY = pad_sequence(devY, batch_first=True)

    trainDL = DataLoader(list(zip(trainX, trainY)), batch_size=1, shuffle=True)

    hSize = 128
    numLayers = 1

    model = RNN(len(trainX[0][0]), len(trainY[0][0]), hSize, numLayers).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(10):
        losses = []
        for i, data in enumerate(trainDL, 0):
            inputs, labels = data
            optimizer.zero_grad()
            hidden = model.initHidden(len(inputs))
            output, hidden = model(inputs, hidden)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        # for i in range(len(trainX)):
        #     hidden = model.initHidden(1)
        #     optimizer.zero_grad()
        #     output, hidden = model(trainX[i].unsqueeze(0), hidden)
        #     loss = criterion(output, trainY[i].unsqueeze(0))
        #     loss.backward()
        #     optimizer.step()
        #     losses.append(loss.item())
        print('----------------------------------')
        print('Epoch:', epoch, 'Loss:', sum(losses) / len(losses))
        with torch.no_grad():
            hidden = model.initHidden(len(devX))
            output, hidden = model(devX, hidden)
            loss = criterion(output, devY)
            print('Dev Loss:', loss.item())
            # correct = 0
            # total = 0
            # for i in range(len(devX)):
            #     hidden = model.initHidden(1)
            #     output, hidden = model(devX[i].unsqueeze(0), hidden)
            #     if output.argmax() == devY[i].argmax():
            #         correct += 1
            #     total += 1
            # print('Dev Accuracy:', correct / total)
    return model

def testRNN(model, testX, testY, device='cpu'):
    with torch.no_grad():
        for i in range(len(testX)):
            testX[i] = torch.tensor(testX[i], dtype=torch.float).to(device)
            testY[i] = torch.tensor(testY[i], dtype=torch.float).to(device)
        testX = pad_sequence(testX, batch_first=True)
        hidden = model.initHidden(len(testX))
        output, hidden = model(testX, hidden)
        correct = 0
        total = 0
        for i in range(len(testY)):
            for j in range(len(testY[i])):
                if output[i][j].argmax() == testY[i][j].argmax():
                    correct += 1
                total += 1
        
        # correct = 0
        # total = 0
        # for i in range(len(testX)):
        #     hidden = model.initHidden(1)
        #     output, hidden = model(testX[i].unsqueeze(0), hidden)
        #     if output.argmax() == testY[i].argmax():
        #         correct += 1
        #     total += 1

        print('Test Accuracy:', correct / total)
        