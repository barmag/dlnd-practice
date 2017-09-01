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
    return cell, initial_state

def build_output(lstm_out, in_size, out_size):
    """Build a fully connected softmax layer, return the softmax output and logits.
    Arguments
    ---------
    lstm_out: Input tensor
    in_size: size of the input tensor
    out_size: size of the softmax layer
    """
    seq_output = tf.concat(lstm_out, axis=1)
    x = tf.reshape([-1, in_size])

    with tf.variable_scope("softmax"):
        softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))

    logits = tf.matmul(x, softmax_w) + softmax_b
    out = tf.nn.softmax(logits, name="predictions")
    return out, logits

def build_loss(logits, targets, lstm_size, num_classes):
    """Calcualte the loss for training
    Arguments
    ---------
    logits: logits from final fully connected layer
    targets: target labels
    lstm_size: number of lstm hidden units
    num_classes: num of classes in targets
    """
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    return loss
