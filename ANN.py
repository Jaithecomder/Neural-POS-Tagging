import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
    
def trainANN(trainX , trainY, pad, contextSize=2, device='cpu'):
    trainX, trainY = dataPrep(trainX, trainY, contextSize, pad)

    trainX = torch.tensor(trainX, dtype=torch.float)
    trainX = torch.flatten(trainX, start_dim=1).to(device)
    trainY = torch.tensor(trainY, dtype=torch.float).to(device)

    print(trainX.shape, trainY.shape)

    model = ANN(len(trainX[0]), len(trainY[0]), 100).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # for epoch in range(100):
