import conllu
import pickle
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from RNN import collate_fn, RNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open("./data/UD_English-Atis/en_atis-ud-train.conllu", "r", encoding="utf-8") as f:
    trainSet = conllu.parse(f.read())
with open("./data/UD_English-Atis/en_atis-ud-dev.conllu", "r", encoding="utf-8") as f:
    devSet = conllu.parse(f.read())

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

dict = {}

with open('RNNTuning.pkl', 'wb') as f:
    pickle.dump(dict, f)

for batchSize in [32, 64, 128]:
    for hSize in [64, 128, 256]:
        for numLayers in [1, 2, 3]:
            for direction in [1, 2]:
                    for lrp in range(2, 5):
                        lr = 10 ** -lrp
                        trainDL = DataLoader(list(zip(newTrainX, newTrainY)), batch_size=batchSize, shuffle=True, collate_fn=lambda x: collate_fn(x, device))

                        model = RNN(len(newTrainX[0][0]), len(newTrainY[0][0]), hSize, numLayers, direction, device=device).to(device)

                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.Adam(model.parameters(), lr=lr)

                        for epoch in range(30):
                            losses = []
                            for i, (x, y) in enumerate(trainDL):
                                optimizer.zero_grad()
                                output = model(x)
                                loss = criterion(output, y)
                                loss.backward()
                                optimizer.step()
                                losses.append(loss.item())
                            print('----------------------------------')
                            print(f'Epoch {epoch+1}/{30} Loss: {sum(losses)/len(losses)}')
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
                            if (epoch + 1) % 10 == 0:
                                with open('RNNTuning.pkl', 'rb') as f:
                                    dict = pickle.load(f)
                                acc = correct / total
                                dict[(batchSize, hSize, numLayers, direction, epoch + 1, lr)] = acc
                                print(f'Batch Size: {batchSize}, Hidden Size: {hSize}, Num Layers: {numLayers}, Direction: {direction}, Epochs: {epoch + 1}, Learning Rate: {lr}')
                                print(f'Accuracy: {acc}')
                                print('----------------------------------')
                                with open('RNNTuning.pkl', 'wb') as f:
                                    pickle.dump(dict, f)