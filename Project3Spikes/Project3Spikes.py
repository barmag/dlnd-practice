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
    #keep_prop = 0.5

    #dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prop)

    cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size) for _ in range(0, num_layers)])
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

def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    embed = get_embed(input_data, vocab_size, embed_dim)
    
    outputs, final_state = build_rnn(cell, embed)
    print(outputs.shape)
    #logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None,
                                               biases_initializer = tf.zeros_initializer(),
                                               weights_initializer= tf.contrib.layers.xavier_initializer_conv2d(
                                                   uniform=True, 
                                                   seed=None, 
                                                   dtype=tf.float32))
    print(logits.shape)
    return logits, final_state

def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    #print(len(int_text))
    batch_seq_size = batch_size*seq_length
    num_batches = len(int_text)//(batch_size*seq_length)
    int_text = int_text[:num_batches*(batch_size*seq_length)]
    print(len(int_text))
    output = np.zeros((num_batches, 2, batch_size, seq_length), dtype=np.int)
    for batch in range(0, num_batches):
        x = [[int_text[x+y] for x in range(0, seq_length)] for y in range(batch*seq_length, (num_batches*batch_seq_size)-(seq_length*(num_batches-batch-1)), seq_length*num_batches)]
        y = [[int_text[x+y+1] if (x+y+1)<len(int_text) else int_text[0] for x in range(0, seq_length)] for y in range(batch*seq_length, (num_batches*batch_seq_size)-(seq_length*(num_batches-batch-1)), seq_length*num_batches)]
        #print(x)
        #print(y)
        output[batch, 0, :, :] = x
        output[batch, 1, :, :] = y

    #print(output.shape)
    return output

def test_get_batches(get_batches):
    with tf.Graph().as_default():
        test_batch_size = 128
        test_seq_length = 5
        test_int_text = list(range(1000*test_seq_length))
        batches = get_batches(test_int_text, test_batch_size, test_seq_length)

        # Check type
        assert isinstance(batches, np.ndarray),\
            'Batches is not a Numpy array'

        # Check shape
        assert batches.shape == (7, 2, 128, 5),\
            'Batches returned wrong shape.  Found {}'.format(batches.shape)

        for x in range(batches.shape[2]):
            assert np.array_equal(batches[0,0,x], np.array(range(x * 35, x * 35 + batches.shape[3]))),\
                'Batches returned wrong contents. For example, input sequence {} in the first batch was {}'.format(x, batches[0,0,x])
            assert np.array_equal(batches[0,1,x], np.array(range(x * 35 + 1, x * 35 + 1 + batches.shape[3]))),\
                'Batches returned wrong contents. For example, target sequence {} in the first batch was {}'.format(x, batches[0,1,x])


        last_seq_target = (test_batch_size-1) * 35 + 31
        last_seq = np.array(range(last_seq_target, last_seq_target+ batches.shape[3]))
        last_seq[-1] = batches[0,0,0,0]

        assert np.array_equal(batches[-1,1,-1], last_seq),\
            'The last target of the last batch should be the first input of the first batch. Found {} but expected {}'.format(batches[-1,1,-1], last_seq)

    print("get batches success")


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
test_input_data_shape = [128, 5]
test_vocab_size = 27

print(test_input_data_shape + [test_vocab_size])

test_get_batches(get_batches)