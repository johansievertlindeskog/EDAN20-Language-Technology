import bz2
import json
import os
import numpy as np
#  import requests
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import confusion_matrix


def count_chars(string, lowercase=True):
    if lowercase:
        string = string.lower()

    char_dict = {}
    tot_nbr_char = len(string)
    for i in range(tot_nbr_char):
        if string[i] in char_dict:
            char_dict[string[i]] = char_dict[string[i]] + 1
        else:
            char_dict[string[i]] = 1

    for key in char_dict:
        char_dict[key] = char_dict[key]/tot_nbr_char

    return char_dict


def count_bigrams(string, lowercase=True):
    if lowercase:
        string = string.lower()

    char_dict = {}
    tot_nbr_bigrams = len(string)-1
    for i in range(tot_nbr_bigrams):
        bigram = string[i] + string[i+1]
        if bigram in char_dict:
            char_dict[bigram] = char_dict[bigram] + 1
        else:
            char_dict[bigram] = 1

    for key in char_dict:
        char_dict[key] = round(char_dict[key]/tot_nbr_bigrams, 5)

    return char_dict


def count_trigrams(string, lc=True):
    if lc == True:
        string = string.lower()

    char_dict = {}
    tot_nbr_trigrams = len(string) - 2
    for i in range(tot_nbr_trigrams):
        trigram = string[i] + string[i+1] + string[i+2]
        if trigram in char_dict:
            char_dict[trigram] = char_dict[trigram] + 1
        else:
            char_dict[trigram] = 1

    for key in char_dict:
        char_dict[key] = round(char_dict[key]/tot_nbr_trigrams, 5)

    return char_dict


if __name__ == '__main__':
    print('languageDetectorLab3')
    dataset = open('sentences.tar/sentences.csv', encoding='utf-8').read().strip()
    dataset = dataset.split('\n')
    dataset[:10]

    dataset = list(map(lambda x: tuple(x.split('\t')), dataset))
    dataset = list(map(lambda x: tuple(map(str.strip, x)), dataset))

    dataset_small = []
    for tp in dataset:
        if tp[1] == 'swe' or tp[1] == 'eng' or tp[1] == 'fra':
            dataset_small.append(tp)

    del dataset

    dataset_small_feat = []
    for tp in dataset_small:
        dataset_small_feat.append((tp[0], tp[1], tp[2], count_chars(tp[2])))

    del dataset_small

    X_cat = []
    for tp in dataset_small_feat:
        X_cat.append(tp[3])

    #  Building the X-matrix
    v = DictVectorizer(sparse=True)
    X = v.fit_transform(X_cat)

    #  Building the y-matrix
    y_cat = []
    for tp in dataset_small_feat:
        y_cat.append(tp[1])

    del dataset_small_feat

    inx2lang = {0: 'fra', 1: 'eng', 2: 'swe'}
    lang2inx = {'fra': 0, 'eng': 1, 'swe': 2}

    y = []
    for lang in y_cat:
        y.append(lang2inx[lang])

    #  Building the model
    clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=5, verbose=True)

    #  Shuffle the indicies so that.......????
    indices = list(range(X.shape[0]))
    np.random.shuffle(indices)
    X = X[indices, :]
    y = np.array(y)[indices]

    #  Splitting the data into training (80%) and validation (20%)
    training_examples = int(X.shape[0] * 0.8)
    X_train = X[:training_examples, :]
    y_train = y[:training_examples]
    X_val = X[training_examples:, :]
    y_val = y[training_examples:]

    # Fitting the model on the training data (this will take several minutes!)
    clf.fit(X_train, y_train)

    # Predicting the values, calculating the accuracy score and confusion matrix
    y_val_pred = clf.predict(X_val)
    accuracy_score(y_val, y_val_pred)

    y_symbols = ['fra', 'eng', 'swe']
    print(classification_report(y_val, y_val_pred, target_names=y_symbols))
    print('Micro F1:', f1_score(y_val, y_val_pred, average='micro'))
    print('Macro F1', f1_score(y_val, y_val_pred, average='macro'))

    confusion_matrix(y_val, y_val_pred)

    #  Testing on some sentences in different languages
    docs = ["Salut les gars !", "Hejsan grabbar!", "Hello guys!", "Hejsan tjejer!"]

    X_cat2 = []
    for sentence in docs:
        X_cat2.append(count_chars(sentence))

    X_test = v.transform(X_cat2)

    pred_languages = clf.predict(X_test)  # print(X_test.shape, X_val.shape, X_train.shape) för att kolla dimensionerna
    y = []
    for lang_code in pred_languages:
        y.append(inx2lang[lang_code])
    pred_languages = y

    print(pred_languages)