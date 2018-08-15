
import numpy as np
import re


def hot_encode(x):
    new_number = x
    one_hot_number = list()
    digit = [0 for _ in range(0,10)]
    digit[new_number] = 1
    one_hot_number.append(digit)
    return one_hot_number
  
  
def one_hot_encoding_char(str_user):
    complete_characters = "abcdefghijklmnopqrstuvwxyz ,./?!"
    characters_index = dict((c,i) for i,c in enumerate(complete_characters))
    sample_text_index = [characters_index[x] for x in str_user]
    sample_text_onehot_encode = []
    for value in sample_text_index:
        char = np.zeros((len(complete_characters),1))
        char[value] = 1
        sample_text_onehot_encode.append(char)
    return sample_text_onehot_encode
  
  
# Helper functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(S):
    return S * (1 - S)

def clipping_nums(num,thresh):
    if num < -thresh:
        num = -thresh
    elif num > thresh:
        num = thresh
    return num

def decimal_to_binary(number):
    r = 0
    q = 0
    bin_num = []
    if number == 1:
        return 1
    else:
        while q != 1:
            q = number // 2
            r = number % 2
            number = q
            bin_num.append(r)
            if q == 1:
                bin_num.append(q)
        return bin_num[::-1]

def vocab_word(document):
    vocab = set(re.findall(r'\w+',document))
    vocab = dict((w,i) for i,w in enumerate(vocab))
    return vocab

def one_hot_encoding_words(vocab, document):
    vocab = set(re.findall(r'\w+',document))
    vocab = dict((w,i) for i,w in enumerate(vocab))
    document_1hot = np.zeros((1,len(vocab)))
    document_words = re.findall(r'\w+',document)
    for i,word in enumerate(vocab):
        if word in document_words:
            document_1hot[:,i] = 1
    return document_1hot

  
