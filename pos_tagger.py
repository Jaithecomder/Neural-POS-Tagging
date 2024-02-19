import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import conllu

trainSet = conllu.parse_incr(open("./data/UD_English-Atis/en_atis-ud-train.conllu", "r", encoding="utf-8"))
testSet = conllu.parse_incr(open("./data/UD_English-Atis/en_atis-ud-test.conllu", "r", encoding="utf-8"))

print(trainSet)