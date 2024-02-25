import conllu
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch

from ANN import dataPrep, ANN

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

hiddenSizes = [(64,),
               (128,),
               (256,),
               (64, 128),
               (128, 256)]

dict = {}

with open('ANNtuning.pkl', 'wb') as f:
    pickle.dump(dict, f)

pad = [0] * len(vocab)
pad[list(vocab).index('<PAD>')] = 1

for contextSize in range(0, 5):
    newTrainX, newTrainY = dataPrep(trainX, trainY, contextSize, pad, device)
    newDevX, newDevY = dataPrep(devX, devY, contextSize, pad, device)
    for hiddenSize in hiddenSizes:
        for activation in ['sigmoid', 'tanh', 'relu']:
            for batchSize in [32, 64, 128]:
                for lrp in range(1, 4):
                    lr = 10 ** -lrp

                    trainDL = DataLoader(list(zip(newTrainX, newTrainY)), batch_size=batchSize, shuffle=True)

                    model = ANN(len(newTrainX[0]), len(newTrainY[0]), hiddenSize, activation, device).to(device)

                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr)

                    for epoch in range(30):
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
                        if (epoch + 1) % 10 == 0:
                            with open('ANNtuning.pkl', 'rb') as f:
                                dict = pickle.load(f)
                            acc = correct / len(output)
                            dict[(contextSize, hiddenSize, activation, batchSize, epoch + 1, lr)] = acc
                            print(f'Context Size: {contextSize}, Hidden Size: {hiddenSize}, Activation: {activation}, Batch Size: {batchSize}, Epochs: {epoch + 1}, Learning Rate: {lr}')
                            print(f'Accuracy: {acc}')
                            print('----------------------------------')
                            with open('ANNtuning.pkl', 'wb') as f:
                                pickle.dump(dict, f)