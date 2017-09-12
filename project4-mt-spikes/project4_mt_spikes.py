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
