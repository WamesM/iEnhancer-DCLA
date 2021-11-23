import itertools
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tqdm as tqdm
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import pickle
import time
import math
import os
import csv
import glob

def load_text_file(file_text):
    with open(file_text) as f:
        lines = f.readlines()
        my_data = [line.strip().upper() for line in lines[1::2]]
        return my_data


def read_test_file(filename):
    text_file = open(filename)
    lines = text_file.readlines()
    m = len(lines) // 5
    my_data = []
    for i in range(m):
        text = lines[i * 5 + 1].strip() + lines[i * 5 + 2].strip() + \
               lines[i * 5 + 3].strip() + lines[i * 5 + 4].strip()
        my_data.append(text.upper())

    return my_data

#NB_WORDS = 65 257 1025 4097 16385 65537
def get_tokenizer():
    f = ['A','C','G','T']
    c = itertools.product(f, f, f, f, f, f, f, f)
    res = []
    for i in c:
        temp = i[0]+i[1]+i[2]+i[3]+i[4]+i[5]+i[6]+i[7]
        res.append(temp)
    res = np.array(res)
    NB_WORDS = 65537
    tokenizer = Tokenizer(num_words=NB_WORDS)
    tokenizer.fit_on_texts(res)
    acgt_index = tokenizer.word_index
    acgt_index['null'] = 0
    return tokenizer


def sentence2word(str_set):
    word_seq = []
    for sr in str_set:
        tmp = []
        for i in range(len(sr)-7):
            if('N' in sr[i:i+8]):
                tmp.append('null')
            else:
                tmp.append(sr[i:i+8])
        word_seq.append(' '.join(tmp))
    return word_seq


def word2num(wordseq, tokenizer, MAX_LEN):
    sequences = tokenizer.texts_to_sequences(wordseq)
    numseq = pad_sequences(sequences, maxlen=MAX_LEN)
    return numseq


def sentence2num(str_set, tokenizer, MAX_LEN):
    wordseq = sentence2word(str_set)
    numseq = word2num(wordseq, tokenizer, MAX_LEN)
    return numseq


def get_data(enhancers):
    tokenizer = get_tokenizer()
    MAX_LEN = 200
    X_en = sentence2num(enhancers, tokenizer, MAX_LEN)
    return X_en


def get_tokenizer_onehot():
    f = ['A','C','G','T']
    res = []
    for i in f:
        res.append(i)
    res = np.array(res)
    NB_WORDS = 5
    tokenizer = Tokenizer(num_words=NB_WORDS)
    tokenizer.fit_on_texts(res)
    acgt_index = tokenizer.word_index
    acgt_index['null'] = 0
    return tokenizer


def sentence2char(str_set):
    char_seq = []
    for sr in str_set:
        tmp = []
        for i in range(len(sr)):
            if('N' in sr[i]):
                tmp.append('null')
            else:
                tmp.append(sr[i])
        char_seq.append(' '.join(tmp))
    return char_seq


def char2num(charseq, tokenizer, MAX_LEN):
    sequences = tokenizer.texts_to_sequences(charseq)
    numseq = pad_sequences(sequences, maxlen=MAX_LEN)
    return numseq


def sentence2num_onehot(str_set, tokenizer, MAX_LEN):
    charseq = sentence2char(str_set)
    numseq = char2num(charseq, tokenizer, MAX_LEN)
    return numseq


def get_data_onehot(enhancers):
    tokenizer = get_tokenizer_onehot()
    MAX_LEN = 200
    X_en = sentence2num_onehot(enhancers, tokenizer, MAX_LEN)


    return X_en

train_strong = load_text_file('D:/pycharm_pro/My-Enhancer-classification/second/train_strong_enhancers.txt')
print("train_strong: ", len(train_strong))
train_weak = load_text_file('D:/pycharm_pro/My-Enhancer-classification/second/train_weak_enhancers.txt')
print("train_weak: ", len(train_weak))
# train_enhancers = train_strong + train_weak
# print("train_enhancers: ", len(train_enhancers))
# train_non_enhancers = load_text_file('D:/pycharm_pro/My-Enhancer-classification/first/train_non_enhancers.txt')
# print("train_non_enhancers: ", len(train_non_enhancers))
train_label_strong = np.ones((len(train_strong), ),dtype=np.float)
train_label_weak = np.zeros((len(train_weak), ),dtype=np.float)
train_data = train_strong + train_weak
train_label = np.hstack((train_label_strong,train_label_weak))
#print(data)

test_strong = read_test_file('D:/pycharm_pro/My-Enhancer-classification/second/test_strong_enhancers.txt')
print("test_strong: ", len(test_strong))
test_weak = read_test_file('D:/pycharm_pro/My-Enhancer-classification/second/test_weak_enhancers.txt')
print("test_weak: ", len(test_weak))
# test_enhancers = test_strong + test_weak
# print("test_enhancers: ", len(test_enhancers))
# test_non_enhancers = read_test_file('D:/pycharm_pro/My-Enhancer-classification/first/test_non_enhancers.txt')
# print("test_non_enhancers: ", len(test_non_enhancers))
test_label_strong = np.ones((len(test_strong), ),dtype=np.float)
test_label_weak = np.zeros((len(test_weak), ),dtype=np.float)
test_data = test_strong + test_weak
test_label = np.hstack((test_label_strong,test_label_weak))


# X_en_tra = get_data(train_data)
# X_en_tes = get_data(test_data)
#
# np.savez('D:/pycharm_pro/My-Enhancer-classification/second/second_index/second_train_enhancers8.npz' ,
#           X_en_tra=X_en_tra, y_tra=train_label)
# np.savez('D:/pycharm_pro/My-Enhancer-classification/second/second_index/second_test_enhancers8.npz' ,
#           X_en_tes=X_en_tes, y_tes=test_label)

X_en_tra_onehot = get_data_onehot(train_data)
X_en_tes_onehot = get_data_onehot(test_data)

np.savez('D:/pycharm_pro/My-Enhancer-classification/second/second_index/second_train_onehot.npz' ,
          X_en_tra=X_en_tra_onehot, y_tra=train_label)
np.savez('D:/pycharm_pro/My-Enhancer-classification/second/second_index/second_test_onehot.npz' ,
          X_en_tes=X_en_tes_onehot, y_tes=test_label)