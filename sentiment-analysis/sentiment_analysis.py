import os
from collections import Counter

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
    