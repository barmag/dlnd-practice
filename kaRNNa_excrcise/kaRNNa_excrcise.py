import numpy as np
import tensorflow as tf

with open('anna.txt', 'r') as file:
    text = file.read()
vocab = sorted(set(text))
#vocab_to_int = {}
#for i, c in enumerate(vocab):
#    vocab_to_int[c] = i
vocab_to_int = {c: i for i, c in enumerate(vocab)}
#encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

encoded = np.array([vocab_to_int[c] for c in text])
print(text[:100])
print(encoded[:100])
