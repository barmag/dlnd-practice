import os
from collections import Counter
import numpy as np

print (os.getcwd())

def process_line(x):
    y = x[:-1]
    return y

def pretty_print_review_and_label(i):
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")

reviews_file = open("reviews.txt")
reviews = list(map(process_line, reviews_file.readlines()))

labels_file = open("labels.txt")
labels = list(map(process_line, labels_file.readlines()))

print(f"reviews are {len(reviews)} and labels {len(labels)}")

positive_count = Counter()
negative_count = Counter()
total_count = Counter()

for i in range(len(reviews)):
    if labels[i] == "positive":
        for word in reviews[i].split(' '):
            positive_count[word] += 1
            total_count[word] += 1
    else:
        for word in reviews[i].split(' '):
            negative_count[word] += 1
            total_count[word] += 1

print(len(positive_count))

positive_negative_ratio = Counter()

for term, count in total_count.most_common():
    if count > 100:
        ratio = positive_count[term] / float(negative_count[term]+1)
        positive_negative_ratio[term] = np.log(ratio)
#print("Pos-to-neg ratio for 'the' = {}".format(positive_negative_ratio["the"]))
#print("Pos-to-neg ratio for 'amazing' = {}".format(positive_negative_ratio["amazing"]))
#print("Pos-to-neg ratio for 'terrible' = {}".format(positive_negative_ratio["terrible"]))

#print("most common positive: ")
#print(positive_negative_ratio.most_common()[0:29])
#print("most common negative: ")
#print(positive_negative_ratio.most_common()[-31:-1])

#unique set of vocabulary
vocab = set(total_count.keys())

layer_0 = np.zeros((1, len(vocab)))

# Create a dictionary of words in the vocabulary mapped to index positions
# (to be used in layer_0)
word2index = {}
for i,word in enumerate(vocab):
    word2index[word] = i

def update_input_layer(review):
    """ Modify the global layer_0 to represent the vector form of review.
    The element at a given index of layer_0 should represent
    how many times the given word occurs in the review.
    Args:
        review(string) - the string of the review
    Returns:
        None
    """
     
    global layer_0
    
    # clear out previous state, reset the layer to be all 0s
    layer_0 *= 0
    
    # count how many times each word is used in the given review and store the results in layer_0 
    for word in review.split(" "):
        layer_0[0][word2index[word]] += 1

def get_target_for_label(label):
    """Convert a label to `0` or `1`.
    Args:
        label(string) - Either "POSITIVE" or "NEGATIVE".
    Returns:
        `0` or `1`.
    """
    if(label == 'positive'):
        return 1
    else:
        return 0

    