from nltk import pos_tag, word_tokenize
import string
import os
import numpy as np
from variables import*

import tensorflow as tf

def remove_punct(line):
    return line.translate(str.maketrans('','',string.punctuation))

def poetry_data():
    sentences = []
    word2idx =  {'START':0, 'END':1}
    current_idx = 2
    for line in open(text_path):
        line = line.strip()
        if line:
            tokens = remove_punct(line.lower()).split(' ')
            sentence = []
            for word in tokens:
                if word not in word2idx:
                    word2idx[word] = current_idx
                    current_idx += 1
                idx = word2idx[word]
                sentence.append(idx)
            sentences.append(sentence)
    print("Data is ready")
    return sentences, word2idx

def init_weight(Mi, Mo):
    return tf.random.normal(shape=(Mi, Mo)) / tf.math.sqrt(tf.constant(Mi+ Mo, dtype=tf.float32))
