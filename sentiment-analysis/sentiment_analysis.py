import os
print (os.getcwd())

def process_line(x):
    y = x[:-1]
    return y

reviews_file = open("reviews.txt")
reviews = list(map(process_line, reviews_file.readlines()))

labels_file = open("labels.txt")
labels = list(map(process_line, labels_file.readlines()))

print(f"reviews are {len(reviews)} and labels {len(labels)}")
