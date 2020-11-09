# import bs4
import os
# import requests
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import time
# from tqdm import tqdm  # och fixa gärna denna också
# import conlleval # måste fixa denna



def read_sentences(file):
    """
    Creates a list of sentences from the corpus
    Each sentence is a string
    :param file:
    :return:
    """
    f = open(file).read().strip()
    sentences = f.split('\n\n')
    return sentences


def split_rows(sentences, column_names):
    """
    Creates a list of sentence where each sentence is a list of lines
    Each line is a dictionary of columns
    :param sentences:
    :param column_names:
    :return:
    """
    new_sentences = []
    for sentence in sentences:
        rows = sentence.split('\n')
        sentence = [dict(zip(column_names, row.split())) for row in rows]
        new_sentences.append(sentence)
    return new_sentences


def count_pos(corpus):
    """
    Computes the part-of-speech distribution
    in a CoNLL 2000 file
    :param corpus:
    :return:
    """
    pos_cnt = {}
    for sentence in corpus:
        for row in sentence:
            if row['pos'] in pos_cnt:
                pos_cnt[row['pos']] += 1
            else:
                pos_cnt[row['pos']] = 1
    return pos_cnt


def chunk_dist(corpus):  # ett corpus
    chunk_dict = {}
    for sentence in corpus:
        for row in sentence:
            if row['pos'] not in chunk_dict:
                chunk_dict[row['pos']] = {}

            if row['chunk'] in chunk_dict[row['pos']]:
                chunk_dict[row['pos']][row['chunk']] += 1
            else:
                chunk_dict[row['pos']][row['chunk']] = 1

    return chunk_dict


def predict(model, corpus):
    bad_pred = corpus
    for i in range(len(bad_pred)):
        for k in range(len(bad_pred[i])):
            bad_pred[i][k]['pchunk'] = model[bad_pred[i][k]['pos']]

    return bad_pred


def eval(predicted):
    """
    Evaluates the predicted chunk accuracy
    :param predicted:
    :return:
    """
    word_cnt = 0
    correct = 0
    for sentence in predicted:
        for row in sentence:
            word_cnt += 1
            if row['chunk'] == row['pchunk']:
                correct += 1
    return correct / word_cnt


def save_results(output_dict, keys, output_file):
    f_out = open(output_file, 'w')
    # We write the word (form), part of speech (pos),
    # gold-standard chunk (chunk), and predicted chunk (pchunk)
    for sentence in output_dict:
        for row in sentence:
            for key in keys:
                f_out.write(row[key] + ' ')
            f_out.write('\n')
        f_out.write('\n')
    f_out.close()
    return


def extract_features_sent_static(sentence, w_size, feature_names):
    """
    Extract the features from one sentence
    returns X and y, where X is a list of dictionaries and
    y is a list of symbols
    :param sentence: string containing the CoNLL structure of a sentence
    :param w_size:
    :return:
    """

    # We pad the sentence to extract the context window more easily
    start = [{'form': 'BOS', 'pos': 'BOS', 'chunk': 'BOS'}]
    end = [{'form': 'EOS', 'pos': 'EOS', 'chunk': 'EOS'}]
    start *= w_size
    end *= w_size
    padded_sentence = start + sentence
    padded_sentence += end

    # We extract the features and the classes
    # X contains is a list of features, where each feature vector is a dictionary
    # y is the list of classes
    X = list()
    y = list()
    for i in range(len(padded_sentence) - 2 * w_size):
        # x is a row of X
        x = list()
        # The words in lower case
        for j in range(2 * w_size + 1):
            x.append(padded_sentence[i + j]['form'].lower())
        # The POS
        for j in range(2 * w_size + 1):
            x.append(padded_sentence[i + j]['pos'])
        # The chunks (Up to the word)
        """
        for j in range(w_size):
            feature_line.append(padded_sentence[i + j]['chunk'])
        """
        # We represent the feature vector as a dictionary
        X.append(dict(zip(feature_names, x)))
        # The classes are stored in a list
        y.append(padded_sentence[i + w_size]['chunk'])
    return X, y


def extract_features_static(sentences, w_size, feature_names):
    """
    Builds X matrix and y vector
    X is a list of dictionaries and y is a list
    :param sentences:
    :param w_size:
    :return:
    """
    X_l = []
    y_l = []
    for sentence in sentences:
        X, y = extract_features_sent_static(sentence, w_size, feature_names)
        X_l.extend(X)
        y_l.extend(y)
    return X_l, y_l


def extract_features_sent_dyn(sentence, w_size, feature_names):
    """
    Extract the features from one sentence
    returns X and y, where X is a list of dictionaries and
    y is a list of symbols
    :param sentence: string containing the CoNLL structure of a sentence
    :param w_size:
    :return:
    """

    # We pad the sentence to extract the context window more easily
    start = [{'form': 'BOS', 'pos': 'BOS', 'chunk': 'BOS'}]
    end = [{'form': 'EOS', 'pos': 'EOS', 'chunk': 'EOS'}]
    start *= w_size
    end *= w_size
    padded_sentence = start + sentence
    padded_sentence += end

    # We extract the features and the classes
    # X contains is a list of features, where each feature vector is a dictionary
    # y is the list of classes
    X = list()
    y = list()
    for i in range(len(padded_sentence) - 2 * w_size):
        # x is a row of X
        x = list()
        # The words in lower case
        for j in range(2 * w_size + 1):
            x.append(padded_sentence[i + j]['form'].lower())
        # The POS
        for j in range(2 * w_size + 1):
            x.append(padded_sentence[i + j]['pos'])
        # The chunks (Up to the word)

        for j in range(w_size):
            x.append(padded_sentence[i + j]['chunk'])

        # We represent the feature vector as a dictionary
        X.append(dict(zip(feature_names, x)))
        # The classes are stored in a list
        y.append(padded_sentence[i + w_size]['chunk'])
    return X, y


def extract_features_dyn(sentences, w_size, feature_names):
    """
    Builds X matrix and y vector
    X is a list of dictionaries and y is a list
    :param sentences:
    :param w_size:
    :return:
    """
    X_l = []
    y_l = []
    for sentence in sentences:
        X, y = extract_features_sent_dyn(sentence, w_size, feature_names)
        X_l.extend(X)
        y_l.extend(y)
    return X_l, y_l


def wikipedia_lookup(ner, base_url='https://en.wikipedia.org/wiki/'):
    try:
        url_en = base_url + ' '.join(ner)
        html_doc = requests.get(url_en).text
        parse_tree = bs4.BeautifulSoup(html_doc, 'html.parser')
        entity_id = parse_tree.find("a", {"accesskey": "g"})['href']
        head_id, entity_id = os.path.split(entity_id)
        return entity_id
    except:
        pass
        # print('Not found in: ', base_url)
    entity_id = 'UNK'
    return entity_id


def ne_ids_en(set_of_entities):
    ne_ids_en = []
    for entity in set_of_entities:
        article = wikipedia_lookup(entity)
        if 'UNK' not in article:
            print(entity)
            ne_ids_en.append(tuple([entity, article]))
    return ne_ids_en


def ne_ids_sv(set_of_entities):
    ne_ids_sv = []
    for entity in set_of_entities:
        article = wikipedia_lookup(entity, 'https://sv.wikipedia.org/wiki/')
        if 'UNK' not in article:
            ne_ids_sv.append(tuple([entity, article]))
    return ne_ids_sv


if __name__ == '__main__':
    print('extractingNounGroupsLab4')

    train_file = 'conll2000/train.txt'
    test_file = 'conll2000/test.txt'

    column_names = ['form', 'pos', 'chunk']
    train_sentences = read_sentences(train_file)
    train_corpus = split_rows(train_sentences, column_names)
    # print(train_corpus[:2])



    # Baseline chunker
    pos_cnt = count_pos(train_corpus)
    # print(pos_cnt)
    chunkDist = chunk_dist(train_corpus)
    # print(chunkDist['NN'])

    pos_chunk = {}
    for pos in chunkDist:
        tp_list = sorted(chunkDist[pos].items(), key=lambda x: x[1], reverse=True)  # sorted returnerar en lista
        pos_chunk[pos] = tp_list[0][0]  # av tuple-object

    # print(pos_chunk['NN'])
    # loading the test corpus to evaluate predict
    test_sentences = read_sentences(test_file)
    test_corpus = split_rows(test_sentences, column_names)
    # print(test_corpus[:1])
    predicted_test_corpus = predict(pos_chunk, test_corpus)
    # print(predicted_test_corpus[:1])
    accuracy = eval(predicted_test_corpus)
    # print(accuracy)



    # The CoNLL evaluation
    keys = ['form', 'pos', 'chunk', 'pchunk']
    save_results(predicted_test_corpus, keys, 'out')
    lines = open('out').read().splitlines()
    res = conlleval.evaluate(lines)
    baseline_score = res['overall']['chunks']['evals']['f1']
    print(baseline_score)

    w_size = 2  # The size of the context window to the left and right of the word
    feature_names = ['word_n2', 'word_n1', 'word', 'word_p1', 'word_p2',
                     'pos_n2', 'pos_n1', 'pos', 'pos_p1', 'pos_p2']
    train_sentences = read_sentences(train_file)
    train_corpus = split_rows(train_sentences, column_names)
    train_corpus[:2]

    X_dict, y = extract_features_static(train_corpus, w_size, feature_names)
    X_dict[:2]
    y[:2]

    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dict)
    classifier = linear_model.LogisticRegression()
    model = classifier.fit(X, y)

    test_sentences = read_sentences(test_file)
    test_corpus = split_rows(test_sentences, column_names)
    test_corpus[:2]

    X_test_dict, y_test = extract_features_static(test_corpus, w_size, feature_names)
    X_test_dict[:2]
    y_test[:2]

    X_test = vec.transform(X_test_dict)  # Possible to add: .toarray()
    y_test_predicted = classifier.predict(X_test)
    y_test_predicted[:2]

    inx = 0
    for sentence in test_corpus:
        for word in sentence:
            word['pchunk'] = y_test_predicted[inx]
            inx += 1

    print(inx)
    print(len(y_test_predicted))
    print(test_corpus[:2])
    save_results(test_corpus, keys, 'out')
    lines = open('out').read().splitlines()
    res = conlleval.evaluate(lines)
    simple_ml_score = res['overall']['chunks']['evals']['f1']
    print(simple_ml_score)



    # Using Machine Learning: Adding all the features from Kudoh and Matsumoto
    feature_names_dyn = ['word_n2', 'word_n1', 'word', 'word_p1', 'word_p2',
                         'pos_n2', 'pos_n1', 'pos', 'pos_p1', 'pos_p2', 'chunk_n2',
                         'chunk_n1']
    train_sentences = read_sentences(train_file)
    train_corpus = split_rows(train_sentences, column_names)
    print(train_corpus[:2])
    X_dict, y = extract_features_dyn(train_corpus, w_size, feature_names_dyn)
    print(X_dict[:30])
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dict)
    classifier = linear_model.LogisticRegression()
    model = classifier.fit(X, y)

    # prediction
    test_sentences = read_sentences(test_file)
    test_corpus = split_rows(test_sentences, column_names)
    test_corpus[:2]
    X_test_dict, y_test = extract_features_static([test_corpus[0]], w_size, feature_names)
    X_test_dict[:2]

    w_size = 2
    y_test_predicted_dyn = []
    for test_sentence in test_corpus:  # när jag fixat tqdm importen funkar "for test_sentence in tqdm(test_corpus):"
        c_2 = 'BOS'                    # för att lägga till en progress bar (detta tar lång tid typ 10 min)
        c_1 = 'BOS'
        [X, y] = extract_features_sent_dyn(test_sentence, w_size, feature_names_dyn)
        for token in X:
            token['chunk_n2'] = c_2
            token['chunk_n1'] = c_1
            X_test = vec.transform([token])  # Possible to add: .toarray()
            y_test_predicted = classifier.predict(X_test)[0]
            c_2 = c_1
            c_1 = y_test_predicted
            y_test_predicted_dyn.append(y_test_predicted)

    # print(y_test_predicted_dyn[:3])

    inx = 0
    for sentence in test_corpus:
        for word in sentence:
            word['pchunk'] = y_test_predicted_dyn[inx]
            inx += 1

    save_results(test_corpus, keys, 'out')

    lines = open('out').read().splitlines()
    res = conlleval.evaluate(lines)
    improved_ml_score = res['overall']['chunks']['evals']['f1']
    print(improved_ml_score)



    # Collecting the entities
    ne_set = set()
    ind_list = list()  # lista med index för träffar av NNP i början av en mening
    for sentence in train_corpus:
        i = 0
        for token in sentence:
            if 'NNP' in token['pos'] and 'B-NP' in token['chunk']:
                ind_list.append(i)
            i = i + 1

        tuple_words = []  # lista av matches som vi ska appenda som ett tuple object
        for index in ind_list:
            k = 1
            tuple_words.append(sentence[index]['form'])
            while len(sentence) - (index + k) > 0:  # för att vi ej ska få index out of bounds
                if 'I-NP' in sentence[index + k]['chunk']:
                    tuple_words.append(sentence[index + k]['form'])
                else:
                    k = len(sentence) + 1  # för att bryta while loopen
                k += 1
            ne_set.add(tuple(tuple_words))
            tuple_words = []
        ind_list = []

    # print(ne_set)
    # print(len(ne_set))
    # print(list(ne_set)[:10])

    # only considering the entities starting with the letter 'K'
    ne_small_set = []  # borde vara ett set MEN det blir ett set pga det vi hämtar från är ett set
    for tp in ne_set:
        if tp[0].startswith('K'):
            ne_small_set.append(tp)

    # print(ne_small_set)

    en_entities = ne_ids_en(ne_small_set)
    sv_entities = ne_ids_sv(ne_small_set)
    # print(en_entities)
    # print(sv_entities)

    # intersect of english and swedish entities
    confirmed_ne_en_sv = []
    en_entities.sort()
    sv_entities.sort()
    for tp in range(len(sv_entities)):
        if sv_entities[tp] in en_entities:
            confirmed_ne_en_sv.append(sv_entities[tp])

    print(confirmed_ne_en_sv)



