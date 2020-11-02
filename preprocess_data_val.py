import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os.path
from gensim.models import FastText


def readData(file_name, max_length):
    data_path = './resources/' + file_name
    if not os.path.isfile(data_path):
        print(file_name + " not found")
        exit()
    with open(data_path, 'rb') as pckl:
        text = pickle.load(pckl)
        for o, doc in enumerate(text):
            text[o] = " ".join(text[o].split()[:max_length])
        return text


def editData(text_file, label_file, max_length, tokenizer):
    text_path = './resources/' + text_file
    if not os.path.isfile(text_path):
        print(text_file + " not found")
        exit()
    with open(text_path, 'rb') as pckl:
        texts = pickle.load(pckl)
    for o, doc in enumerate(texts):
        texts[o] = " ".join(texts[o].split()[:max_length])
    sequences = tokenizer.texts_to_sequences(texts)
    del texts
    data = pad_sequences(sequences, maxlen=max_length,
                         padding='post', truncating='post')
    del sequences
    label_path = './resources/' + label_file
    if not os.path.isfile(label_path):
        print(label_file + " not found")
        exit()
    with open(label_path, 'rb') as pckl:
        labels = pickle.load(pckl)
    data = data.astype(np.uint16)
    return data, labels


def preprocess(max_length, max_num_words):
    
    texts_tokenize = readData('train_texts.pkl', max_length)
    print("Tokenizing data ..")
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(texts_tokenize)
    print("Tokenization and fitting done!")
    print("Loading data ...")

    x_val, y_val = editData('val_texts.pkl', 'val_labels.pkl',
                            max_length, tokenizer)
    print("Test Data loaded")
    
    return x_val, y_val
