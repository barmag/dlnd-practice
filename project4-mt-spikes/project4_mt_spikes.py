import tensorflow as tf

def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    source_sentences = source_text.split('\n')
    print(source_sentences)
    source_id_text = [[source_vocab_to_int[word] for word in sentence.split()] for sentence in source_sentences]
    print(source_id_text)
    target_sentences = [sentence + " <EOS>" for sentence in target_text.split('\n')]
    target_id_text = [[target_vocab_to_int[word] for word in sentence.split()] for sentence in target_sentences]
    
    return source_id_text, target_id_text

def model_inputs():
    """
    Create TF Placeholders for input, targets, learning rate, and lengths of source and target sequences.
    :return: Tuple (input, targets, learning rate, keep probability, target sequence length,
    max target sequence length, source sequence length)
    """
    inputs = tf.placeholder(tf.int32, [None, None], name="input")
    targets = tf.placeholder(tf.int32, [None, None], name="target")

    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    target_seq_length = tf.placeholder(tf.int32, (None,), name="target_sequence_length")
    max_target_length = tf.reduce_max(target_seq_length, name="max_target_length")
    source_seq_length = tf.placeholder(tf.int32, (None,), name="source_sequence_length")

    return inputs, targets, learning_rate, keep_prob, target_seq_length, max_target_length, source_seq_length

def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for encoding
    :param target_data: Target Placehoder
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """
    decoder_input = tf.strided_slice(target_data, [0,0], [batch_size, -1], [1,1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int['<GO>']), decoder_input], 1)
    return decoder_input

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, 
                   source_sequence_length, source_vocab_size, 
                   encoding_embedding_size):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :param source_sequence_length: a list of the lengths of each sequence in the batch
    :param source_vocab_size: vocabulary size of source data
    :param encoding_embedding_size: embedding size of source data
    :return: tuple (RNN output, RNN state)
    """
    embed = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, encoding_embedding_size)

    def lstm_cell(size):
        cell = tf.contrib.rnn.LSTMCell(size,
                                       initializer=tf.random_uniform_initializer(-0.1,0.1,seed=7))
        dropout = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
        return dropout
    rnn_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(rnn_size) for _ in range(0, num_layers)])

    rnn_out, rnn_state = tf.nn.dynamic_rnn(rnn_cell, embed, source_sequence_length, dtype=tf.float32)
    return rnn_out, rnn_state

print("tf loaded!")