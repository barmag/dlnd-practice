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

def get_batches(arr, n_seqs, n_steps):
    """
    Create a generator that returns batches of size
    n_seqs, x n_steps from array
    Arguments
    ---------
    arr: Array of int represented characters to get batches from
    n_seqs: Batch size, the number of sequences per batch
    n_steps: number of sequence steps per batch
    """
    # number of chars per batch and number of batches
    chars_per_batch = n_seqs * n_steps
    n_batches = len(arr)//chars_per_batch

    # trim array to exactly number of bataches (no padding)
    arr = arr[:n_batches*chars_per_batch]

    # reshape into number of seqs rows
    arr = np.reshape(arr, (n_seqs, -1))
    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:+n_steps]
        # labels (x shifted by 1)
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y

batches = get_batches(encoded, 10, 50)
x, y = next(batches)

# build network
def build_inputs(batch_size, num_steps):
    '''Define placeholders for inputs, labels, and dropout
    Arguments
    ---------
    batch_size: Batch size
    num_steps: number of sequence steps per batch
    '''
    inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name="inputs")
    labels = tf.placeholder(tf.int32, [batch_size, num_steps], name="labels")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    return input, labels, keep_prob

# build lstm layer
def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    """Build LSTM cell
    Argumets
    --------
    lstm_size: size of hidden layers in cell
    num_layers: number of LSTM layers
    batch_size: batch size
    keep_prob: dropout keep probability
    """
    def build_cell(lstm_size, keep_prob):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

        return drop

    # stack multiple lstm cells, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)