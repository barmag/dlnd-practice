import numpy as np
import tensorflow as tf
import time

with open('anna.txt', 'r') as file:
    text = file.read()
vocab = sorted(set(text))
#vocab_to_int = {}
#for i, c in enumerate(vocab):
#    vocab_to_int[c] = i
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
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
        x = arr[:, n:n+n_steps]
        # labels (x shifted by 1)
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y

#batches = get_batches(encoded, 10, 50)
#x, y = next(batches)

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
    return inputs, labels, keep_prob

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
    x = tf.reshape(seq_output, [-1, in_size])

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

# optimizer
def build_optimizer(loss, learning_rate, grad_clip):
    """Build optimizer for training, using gradient clip
    Argumnets
    ---------
    loss: network loss
    learning_rate: network learning rate
    grad_clip: upper bound to clip gradients to prevent value explosion
    """
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    return optimizer

class CharRNN:
    def __init__(self, num_classes, batch_size=64, num_steps=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False):
        
        if sampling == True:
            batch_size, num_steps = 1,1
        
        tf.reset_default_graph()
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)

        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)

        x_one_hot = tf.one_hot(self.inputs, num_classes)
        output, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state

        self.prediction, self.logits = build_output(output, lstm_size, num_classes)

        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)

# hyper parameters
batch_size = 100
num_steps = 100
lstm_size = 512
num_layers = 2
learning_rate = 0.001
keep_prob = 0.5

# training
def train_network():
    epochs = 20
    save_every_n = 200
    
    model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps, lstm_size=lstm_size
                    , num_layers=num_layers, learning_rate=learning_rate)
    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'checkpoints/i3960_l512.ckpt')
        counter = 0
        for e in range(epochs):
            new_state = sess.run(model.initial_state)
            loss = 0
            for x,y in get_batches(encoded, batch_size, num_steps):
                counter += 1
                start = time.time()
                feed_dict = {model.inputs: x, model.targets: y, model.keep_prob: keep_prob,
                             model.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([model.loss, model.final_state, model.optimizer]
                                                    , feed_dict=feed_dict)
                end = time.time()
                print('Epoch: {}/{}... '.format(e+1, epochs),
                      'Training Step: {}... '.format(counter),
                      'Training loss: {:.4f}... '.format(batch_loss),
                      '{:.4f} sec/batch'.format((end-start)))
                if (counter % save_every_n == 0):
                    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
        saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
    return epochs, model, save_every_n, saver

# epochs, model, save_every_n, saver = train_network()

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    samples = [c for c in prime]
    model = CharRNN(len(vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0,0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

        c = pick_top_n(preds, len(vocab))
        samples.append(int_to_vocab[c])

        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])
        
    return ''.join(samples)

checkpoint = tf.train.latest_checkpoint('checkpoints')
samp = sample(checkpoint, 2000, lstm_size, len(vocab), prime="The ")
print(samp)
