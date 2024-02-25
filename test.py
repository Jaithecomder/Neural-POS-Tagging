import pickle

with open('RNNTuning.pkl', 'rb') as f:
    dict = pickle.load(f)

print(dict)