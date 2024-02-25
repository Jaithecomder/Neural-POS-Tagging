import conllu
import pickle
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from ANN import dataPrep, ANN

with open('ANNTuning.pkl', 'rb') as f:
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

pad = [0] * len(vocab)
pad[list(vocab).index('<PAD>')] = 1

for item in best3:
    (contextSize, hiddenSize, activation, batchSize, epochs, lr) = item[0]

    newTrainX, newTrainY = dataPrep(trainX, trainY, contextSize, pad, device)
    newDevX, newDevY = dataPrep(devX, devY, contextSize, pad, device)
    newTestX, newTestY = dataPrep(testX, testY, contextSize, pad, device)

    trainDL = DataLoader(list(zip(newTrainX, newTrainY)), batch_size=batchSize, shuffle=True)

    model = ANN(len(newTrainX[0]), len(newTrainY[0]), hiddenSize, activation, device, contextSize).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    devAcc = []

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
            output = model(newDevX)
            correct = 0
            for i in range(len(output)):
                if torch.argmax(output[i]) == torch.argmax(newDevY[i]):
                    correct += 1
            print('Dev Accuracy:', correct / len(output))
            devAcc.append(correct / len(output))
    torch.save(model, f'./ANNSaves/ANNTuning_{contextSize}_{hiddenSize}_{activation}_{batchSize}_{epochs}_{lr}.pt')
    plt.figure()
    plt.plot(devAcc)
    plt.title(f'Context Size: {contextSize}, Hidden Size: {hiddenSize}, Activation: {activation}, Batch Size: {batchSize}, Epochs: {epochs}, Learning Rate: {lr}')
    plt.xlabel('Epochs')
    plt.ylabel('Dev Accuracy')
    plt.savefig(f'./ANNSaves/ANNTuning_{contextSize}_{hiddenSize}_{activation}_{batchSize}_{epochs}_{lr}.png')

    with torch.no_grad():
        output = model(newDevX)
        devCMat = 100 * confusion_matrix(newDevY.cpu().numpy().argmax(axis=1), output.cpu().numpy().argmax(axis=1), labels=range(17), normalize="true")
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
        plt.title(f'Context Size: {contextSize}, Hidden Size: {hiddenSize}, Activation: {activation}, Batch Size: {batchSize}, Epochs: {epochs}, Learning Rate: {lr}')
        plt.savefig(f'./ANNSaves/ANNTuning_{contextSize}_{hiddenSize}_{activation}_{batchSize}_{epochs}_{lr}_Dev_CMat.png')

        with open(f'./ANNSaves/ANNTuning_{contextSize}_{hiddenSize}_{activation}_{batchSize}_{epochs}_{lr}.txt', 'w') as f:
            f.write(f'Dev Accuracy: {accuracy_score(newDevY.cpu().numpy().argmax(axis=1), output.cpu().numpy().argmax(axis=1))}\n')
            f.write(f'Dev Precision: {precision_score(newDevY.cpu().numpy().argmax(axis=1), output.cpu().numpy().argmax(axis=1), average="macro", labels=range(17), zero_division=0)}\n')
            f.write(f'Dev Recall: {recall_score(newDevY.cpu().numpy().argmax(axis=1), output.cpu().numpy().argmax(axis=1), average="macro", labels=range(17), zero_division=0)}\n')
            f.write(f'Dev micro F1 Score: {f1_score(newDevY.cpu().numpy().argmax(axis=1), output.cpu().numpy().argmax(axis=1), average="micro", labels=range(17), zero_division=0)}\n')
            f.write(f'Dev macro F1 Score: {f1_score(newDevY.cpu().numpy().argmax(axis=1), output.cpu().numpy().argmax(axis=1), average="macro", labels=range(17), zero_division=0)}\n')

        output = model(newTestX)
        testCMat = 100 * confusion_matrix(newTestY.cpu().numpy().argmax(axis=1), output.cpu().numpy().argmax(axis=1), labels=range(17), normalize="true")
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
        plt.title(f'Context Size: {contextSize}, Hidden Size: {hiddenSize}, Activation: {activation}, Batch Size: {batchSize}, Epochs: {epochs}, Learning Rate: {lr}')
        plt.savefig(f'./ANNSaves/ANNTuning_{contextSize}_{hiddenSize}_{activation}_{batchSize}_{epochs}_{lr}_Test_CMat.png')

        with open(f'./ANNSaves/ANNTuning_{contextSize}_{hiddenSize}_{activation}_{batchSize}_{epochs}_{lr}.txt', 'w') as f:
            f.write(f'Test Accuracy: {accuracy_score(newTestY.cpu().numpy().argmax(axis=1), output.cpu().numpy().argmax(axis=1))}\n')
            f.write(f'Test Precision: {precision_score(newTestY.cpu().numpy().argmax(axis=1), output.cpu().numpy().argmax(axis=1), average="macro", labels=range(17), zero_division=0)}\n')
            f.write(f'Test Recall: {recall_score(newTestY.cpu().numpy().argmax(axis=1), output.cpu().numpy().argmax(axis=1), average="macro", labels=range(17), zero_division=0)}\n')
            f.write(f'Test micro F1 Score: {f1_score(newTestY.cpu().numpy().argmax(axis=1), output.cpu().numpy().argmax(axis=1), average="micro", labels=range(17), zero_division=0)}\n')
            f.write(f'Test macro F1 Score: {f1_score(newTestY.cpu().numpy().argmax(axis=1), output.cpu().numpy().argmax(axis=1), average="macro", labels=range(17), zero_division=0)}\n')