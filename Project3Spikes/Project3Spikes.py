from collections import Counter
import numpy as np
import string

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
