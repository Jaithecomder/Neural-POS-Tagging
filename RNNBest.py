import conllu
import pickle
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from RNN import collate_fn, RNN

with open('RNNTuning.pkl', 'rb') as f:
    dict = pickle.load(f)

dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)

best3 = dict[:3]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open("./data/UD_English-Atis/en_atis-ud-train.conllu", "r", encoding="utf-8") as f:
    trainSet = conllu.parse(f.read())
with open("./data/UD_English-Atis/en_atis-ud-dev.conllu", "r", encoding="utf-8") as f:
    devSet = conllu.parse(f.read())
with open("./data/UD_English-Atis/en_atis-ud-test.conllu", "r", encoding="utf-8") as f:
    testSet = conllu.parse(f.read())

vocab = set()
pos = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

for sentence in trainSet:
    for word in sentence:
        vocab.add(word['form'])

vocab.add('<OOV>')
vocab.add('<PAD>')

def oneHotEncode(sentenceSet, vocab, pos):
    X = []
    Y = []

    for sentence in sentenceSet:
        x = []
        y = []
        for word in sentence:
            wx = [0] * len(vocab)
            wy = [0] * len(pos)
            if word['form'] in vocab:
                wx[list(vocab).index(word['form'])] = 1
            else:
                wx[list(vocab).index('<OOV>')] = 1
            wy[list(pos).index(word['upos'])] = 1
            x.append(wx)
            y.append(wy)
        X.append(x)
        Y.append(y)

    return X, Y

trainX, trainY = oneHotEncode(trainSet, vocab, pos)
devX, devY = oneHotEncode(devSet, vocab, pos)
testX, testY = oneHotEncode(testSet, vocab, pos)

newTrainX = []
newTrainY = []
newDevX = []
newDevY = []
newTestX = []
newTestY = []
for i in range(len(trainX)):
    newTrainX.append(torch.tensor(trainX[i], dtype=torch.float).to(device))
    newTrainY.append(torch.tensor(trainY[i], dtype=torch.float).to(device))
for i in range(len(devX)):
    newDevX.append(torch.tensor(devX[i], dtype=torch.float).to(device))
    newDevY.append(torch.tensor(devY[i], dtype=torch.float).to(device))
for i in range(len(testX)):
    newTestX.append(torch.tensor(testX[i], dtype=torch.float).to(device))
    newTestY.append(torch.tensor(testY[i], dtype=torch.float).to(device))

for item in best3:
    (batchSize, hSize, numLayers, direction, epochs, lr) = item[0]

    trainDL = DataLoader(list(zip(newTrainX, newTrainY)), batch_size=batchSize, shuffle=True, collate_fn=lambda x: collate_fn(x, device))

    model = RNN(len(newTrainX[0][0]), len(newTrainY[0][0]), hSize, numLayers, direction, device=device).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    devAcc = []

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
            devAcc.append(correct / total)
    torch.save(model, f'./RNNsaves/RNNTuning_{batchSize}_{hSize}_{numLayers}_{direction}_{lr}.pt')
    plt.figure()
    plt.plot(devAcc)
    plt.title(f'Batch Size: {batchSize}, Hidden Size: {hSize}, Num Layers: {numLayers}, Direction: {direction}, Learning Rate: {lr}')
    plt.xlabel('Epochs')
    plt.ylabel('Dev Accuracy')
    plt.savefig(f'./RNNsaves/RNNTuning_{batchSize}_{hSize}_{numLayers}_{direction}_{lr}.png')

    with torch.no_grad():
        outputs = torch.tensor([]).to(device)
        for i in range(len(newDevX)):
            outputs = torch.cat((outputs, model(newDevX[i])))
        flatDevY = torch.tensor([]).to(device)
        for i in range(len(newDevY)):
            flatDevY = torch.cat((flatDevY, newDevY[i]))

        devCMat = 100 * confusion_matrix(flatDevY.cpu().numpy().argmax(axis=1), outputs.cpu().numpy().argmax(axis=1), labels=range(17), normalize="true")
        plt.figure(figsize=(16, 16))
        plt.imshow(devCMat, cmap='bwr', interpolation='nearest', vmin=0, vmax=100)
        plt.colorbar()
        for i in range(17):
            for j in range(17):
                plt.text(j, i, '{:.2f}'.format(devCMat[i, j]), ha='center', va='center', color='white')
        plt.xticks(range(17), pos, rotation=90)
        plt.yticks(range(17), pos)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Batch Size: {batchSize}, Hidden Size: {hSize}, Num Layers: {numLayers}, Direction: {direction}, Learning Rate: {lr}')
        plt.savefig(f'./RNNsaves/RNNTuning_{batchSize}_{hSize}_{numLayers}_{direction}_{lr}_Dev_CMat.png')
        
        with open(f'./RNNsaves/RNNTuning_{batchSize}_{hSize}_{numLayers}_{direction}_{lr}.txt', 'w') as f:
            f.write(f'Dev Accuracy: {accuracy_score(flatDevY.cpu().numpy().argmax(axis=1), outputs.cpu().numpy().argmax(axis=1))}\n')
            f.write(f'Dev Precision: {precision_score(flatDevY.cpu().numpy().argmax(axis=1), outputs.cpu().numpy().argmax(axis=1), average="macro", labels=range(17), zero_division=0)}\n')
            f.write(f'Dev Recall: {recall_score(flatDevY.cpu().numpy().argmax(axis=1), outputs.cpu().numpy().argmax(axis=1), average="macro", labels=range(17), zero_division=0)}\n')
            f.write(f'Dev micro F1 Score: {f1_score(flatDevY.cpu().numpy().argmax(axis=1), outputs.cpu().numpy().argmax(axis=1), average="micro", labels=range(17), zero_division=0)}\n')
            f.write(f'Dev macro F1 Score: {f1_score(flatDevY.cpu().numpy().argmax(axis=1), outputs.cpu().numpy().argmax(axis=1), average="macro", labels=range(17), zero_division=0)}\n')
        
        outputs = torch.tensor([]).to(device)
        for i in range(len(newTestX)):
            outputs = torch.cat((outputs, model(newTestX[i])))
        flatTestY = torch.tensor([]).to(device)
        for i in range(len(newTestY)):
            flatTestY = torch.cat((flatTestY, newTestY[i]))
        
        testCMat = 100 * confusion_matrix(flatTestY.cpu().numpy().argmax(axis=1), outputs.cpu().numpy().argmax(axis=1), labels=range(17), normalize="true")
        plt.figure(figsize=(16, 16))
        plt.imshow(testCMat, cmap='bwr', interpolation='nearest', vmin=0, vmax=100)
        plt.colorbar()
        for i in range(17):
            for j in range(17):
                plt.text(j, i, '{:.2f}'.format(testCMat[i, j]), ha='center', va='center', color='white')
        plt.xticks(range(17), pos, rotation=90)
        plt.yticks(range(17), pos)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Batch Size: {batchSize}, Hidden Size: {hSize}, Num Layers: {numLayers}, Direction: {direction}, Learning Rate: {lr}')
        plt.savefig(f'./RNNsaves/RNNTuning_{batchSize}_{hSize}_{numLayers}_{direction}_{lr}_Test_CMat.png')

        with open(f'./RNNsaves/RNNTuning_{batchSize}_{hSize}_{numLayers}_{direction}_{lr}.txt', 'a') as f:
            f.write(f'Test Accuracy: {accuracy_score(flatTestY.cpu().numpy().argmax(axis=1), outputs.cpu().numpy().argmax(axis=1))}\n')
            f.write(f'Test Precision: {precision_score(flatTestY.cpu().numpy().argmax(axis=1), outputs.cpu().numpy().argmax(axis=1), average="macro", labels=range(17), zero_division=0)}\n')
            f.write(f'Test Recall: {recall_score(flatTestY.cpu().numpy().argmax(axis=1), outputs.cpu().numpy().argmax(axis=1), average="macro", labels=range(17), zero_division=0)}\n')
            f.write(f'Test micro F1 Score: {f1_score(flatTestY.cpu().numpy().argmax(axis=1), outputs.cpu().numpy().argmax(axis=1), average="micro", labels=range(17), zero_division=0)}\n')
            f.write(f'Test macro F1 Score: {f1_score(flatTestY.cpu().numpy().argmax(axis=1), outputs.cpu().numpy().argmax(axis=1), average="macro", labels=range(17), zero_division=0)}\n')