import re
import scipy.spatial.distance
import numpy as np

file_obj = open('sentences.txt', 'r')
sentences_orig = list(file_obj)
sentences_lower = []

for sentence in sentences_orig:
    sentences_lower.append(sentence.lower())

n = len(sentences_lower)
words = {}
sentences_split = []
for sentence_l in sentences_lower:
    row = []
    for word in re.split('[^a-z]', sentence_l):
        word_strip = word.strip()
        if len(word_strip) > 0:
            row.append(word_strip)
    sentences_split.append(row)

idx = 0
for sentence_s_arr in sentences_split:
    for w in sentence_s_arr:
        if w in words:
            continue;
        words[w] = idx
        idx = idx + 1

sentences_freqs = []
d = len(words)
for sentence_s_arr in sentences_split:
    i = 0
    freqs_row = [0] * d
    for dict_w in sentence_s_arr:
        word_idx = words[dict_w]
        freqs_row[word_idx] = freqs_row[word_idx] + 1;
    sentences_freqs.append(freqs_row)

base = sentences_freqs[0]
distances = []
for i in range(0, n):
    dist = scipy.spatial.distance.cosine(base, sentences_freqs[i])
    distances.append(dist)

top2 = np.argsort(distances)[1:3]
print(top2)
file_obj = open('submission-1.txt', 'w')
file_obj.write(str(top2[0]) + ' ' + str(top2[1]))
file_obj.close()