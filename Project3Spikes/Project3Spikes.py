from collections import Counter
import numpy as np
import string
import tensorflow as tf

def create_lookup_tables(text):
    # normalize text
    local_text = text.lower()
    local_text = local_text.split()

    counter = Counter(local_text)
    print(counter.most_common(20))
    vocab_to_int, int_to_vocab = {}, {}
    vocab_list = (zip(counter, range(0, len(counter))))
    
    vocab_to_int = {word: i for word, i in vocab_list}
    int_to_vocab = dict(zip(vocab_to_int.values(), vocab_to_int.keys()))
    return vocab_to_int, int_to_vocab
def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    punctuation_dict = {
        '.': 'period', ',': 'comma', '"': 'quotation_mark', 
        ';': 'semicolon', '!': 'exclamation_mark', '?': 'question_mark',
        '(': 'left_parentheses', ')': 'right_parentheses', '--': 'dash', '\n': 'return'
    }
    return punctuation_dict

def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    inputs = tf.placeholder(tf.int32, shape=(None, None), name='input')
    targets = tf.placeholder(tf.int32, shape=(None, None))
    learning_rate = tf.placeholder(tf.float32)
    return inputs, targets, learning_rate

def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    num_layers = 1
    keep_prop = 0.5
    
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)

    dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prop)

    cell = tf.contrib.rnn.MultiRNNCell([dropout]*num_layers)
    initial_state = tf.identity(cell.zero_state(batch_size, tf.float32), 'initial_state')
    return cell, initial_state

def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embeddings_matrix = tf.random_uniform([vocab_size, embed_dim], -1.0, 1.0)

    embed = tf.nn.embedding_lookup(embeddings_matrix, input_data)
    return embed

def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(final_state, 'final_state')
    return outputs, final_state

text = '''
        Moe_Szyslak Moe's Tavern Where the elite meet to drink
        Bart_Simpson Eh yeah hello is Mike there Last name Rotch
        Moe_Szyslak Hold on I'll check Mike Rotch Mike Rotch Hey has anybody seen Mike Rotch lately
        Moe_Szyslak Listen you little puke One of these days I'm gonna catch you and I'm gonna carve my name on your back with an ice pick
        Moe_Szyslak Whats the matter Homer You're not your normal effervescent self
        Homer_Simpson I got my problems Moe Give me another one
        Moe_Szyslak Homer hey you should not drink to forget your problems
        Barney_Gumble Yeah you should only drink to enhance your social skills'''

# create_lookup_tables(text)
