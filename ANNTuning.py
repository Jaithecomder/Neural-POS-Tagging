import conllu
import torch
import pickle

from ANN import trainANN, testANN

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
               (512,),
               (64, 128),
               (128, 256),
               (256, 512),
               (64, 256),
               (128, 512),
               (64, 128, 256),
               (128, 256, 512),
               (64, 128, 256, 512)]

dict = {}
for contextSize in range(0, 5):
    for hiddenSize in hiddenSizes:
        for activation in ['sigmoid', 'tanh', 'relu']:
            for batchSize in [32, 64, 128]:
                for epochs in range(10, 41, 10):
                    for lrp in range(1, 5):
                        lr = 10 ** -lrp
                        pad = [0] * len(vocab)
                        pad[list(vocab).index('<PAD>')] = 1
                        model = trainANN(trainX, trainY, devX, devY, pad, contextSize=contextSize,
                                         lr=lr, hiddenSizes=hiddenSize, activation=activation,
                                         batchSize=batchSize, epochs=epochs, device=device)
                        acc = testANN(model, devX, devY, pad, contextSize=contextSize, device=device)
                        dict[(contextSize, hiddenSize, activation, batchSize, epochs, lr)] = acc
                        print(f'Context Size: {contextSize}, Hidden Size: {hiddenSize}, Activation: {activation}, Batch Size: {batchSize}, Epochs: {epochs}, Learning Rate: {lr}')
                        print(f'Accuracy: {acc}')
                        print('----------------------------------')

with open('ANNtuning.pkl', 'wb') as f:
    pickle.dump(dict, f)