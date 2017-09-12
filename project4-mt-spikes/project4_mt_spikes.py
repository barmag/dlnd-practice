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

print("tf loaded!")