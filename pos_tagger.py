import conllu
import torch
import sys
import os

from ANN import trainANN
from RNN import trainRNN

modelc = '-f'
if len(sys.argv) > 1:
    modelc = sys.argv[1]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sentence = input("Enter a sentence: ")
tokens = sentence.split()

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

model = None

if modelc == '-f':
    if os.path.exists('ANNBest.pt'):
        model = torch.load('ANNBest.pt')
    else:
        context = 2
        pad = [0] * len(vocab)
        pad[list(vocab).index('<PAD>')] = 1
        model = trainANN(trainX, trainY, devX, devY, pad, contextSize=context, device=device)
        
elif modelc == '-r':
    if os.path.exists('RNNBest.pt'):
        model = torch.load('RNNBest.pt')
    else:
        model = trainRNN(trainX, trainY, devX, devY, device=device)

tokens1h = []
for token in tokens:
    wx = [0] * len(vocab)
    if token in vocab:
        wx[list(vocab).index(token)] = 1
    else:
        wx[list(vocab).index('<OOV>')] = 1
    tokens1h.append(wx)

preds = []
if modelc == '-f':
    contextSize = model.contextSize
    pad = [0] * len(vocab)
    pad[list(vocab).index('<PAD>')] = 1
    X = []
    for i in range(len(tokens1h)):
        x = []
        for j in range(i - contextSize, i + contextSize + 1):
            if j < 0 or j >= len(tokens1h):
                x.append(pad)
            else:
                x.append(tokens1h[j])
        X.append(x)
    X = torch.tensor(X, dtype=torch.float)
    X = torch.flatten(X, start_dim=1).to(device)
    preds = model(X)
elif modelc == '-r':
    for token in tokens1h:
        x = torch.tensor([token], dtype=torch.float).to(device)
        pred = model(x)
        preds.append(pred)

for i in range(len(tokens)):
    print(tokens[i], pos[preds[i].argmax()])