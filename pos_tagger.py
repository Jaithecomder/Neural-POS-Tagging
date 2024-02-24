import conllu
import torch
import sys

from ANN import trainANN, testANN
from RNN2 import trainRNN, testRNN

model = '-f'
if len(sys.argv) > 1:
    model = sys.argv[1]

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

if model == '-f':
    context = 2
    pad = [0] * len(vocab)
    pad[list(vocab).index('<PAD>')] = 1
    model = trainANN(trainX, trainY, devX, devY, pad, contextSize=context, device=device)
    acc = testANN(model, testX, testY, pad, context, device)
    print(f'Accuracy: {acc}')

elif model == '-r':
    model = trainRNN(trainX, trainY, devX, devY, device)
    testRNN(model, testX, testY, device)