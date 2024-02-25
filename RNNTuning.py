import conllu
import torch
import pickle

from RNN import trainRNN, testRNN

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

dict = {}

with open('RNNTuning.pkl', 'wb') as f:
    pickle.dump(dict, f)

for batchSize in [32, 64, 128]:
    for hSize in [64, 128, 256]:
        for numLayers in [1, 2, 3]:
            for direction in [1, 2]:
                for epochs in range(10, 31, 10):
                    for lrp in range(1, 4):
                        with open('RNNTuning.pkl', 'rb') as f:
                            dict = pickle.load(f)
                        lr = 10 ** -lrp
                        model = trainRNN(trainX, trainY, devX, devY, batchSize=batchSize,
                                         hSize=hSize, numLayers=numLayers, direction=direction,
                                         epochs=epochs, lr=lr, device=device)
                        acc = testRNN(model, devX, devY, device=device)
                        dict[(batchSize, hSize, numLayers, direction, epochs, lrp)] = acc
                        print(f'Batch Size: {batchSize}, Hidden Size: {hSize}, Num Layers: {numLayers}, Direction: {direction}, Epochs: {epochs}, Learning Rate: {lr}')
                        print(f'Accuracy: {acc}')
                        print('----------------------------------')
                        with open('RNNTuning.pkl', 'wb') as f:
                            pickle.dump(dict, f)