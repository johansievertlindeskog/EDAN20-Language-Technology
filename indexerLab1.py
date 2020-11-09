import math
import os
#  import pickle
import regex as re
import numpy as np


def tokenize(text):  # file_text är en string, inte en lista
    text = text.lower().strip()  # kanske skall göras manuellt när man anropar tokenize?
    list = []

    for m in re.finditer(r'\p{L}+', text):  # r'\p{L}+' = unicode commando for finding sequences of letters, i.e. words
        list.append(m)

    return list  # index 0-2 på första, räknas mellanslag? för sista visar 56-61 oavsett om punkt är med eller ej


def text_to_idx(words):  # words är en list av regex-match objects
    dictionary = {}
    for w in words:
        if w.group(0) in dictionary:                      # kollar dubbletter
            dictionary[w.group(0)].append(w.start())      # dictionary[w.group(0)] är en lista som jag appendar till
        else:
            dictionary[w.group(0)] = [w.start()]          # skapar en ny lista och lägger till startindex
    return dictionary  # notera att den returnerar ett dictionary


def get_files(dir, suffix):
    """
    Returns all the files in a folder ending with suffix
    :param dir: # directory
    :param suffix: # ex. 'txt'
    :return: the list of file names
    """

    files = []
    for file in os.listdir(dir):
        if file.endswith(suffix):
            files.append(file)
    return files


def master_index(txt_files):  # tar in en lista av textfiler, loopar igenom

    master_idx = {}
    text_idx = {}

    for f in txt_files:
        file = open('Selma/' + f, encoding='utf-8')
        file_text = file.read()
        tokens = tokenize(file_text)  # file_text är en string, inte en lista
        idx = text_to_idx(tokens)  # tokens är en lista av ord
        text_idx[f] = idx  # mappar filnamnet till index

        for word in text_idx[f]:  # word i detta fall blir nycklarna till text_idx[f]
            if word in master_idx:
                master_idx[word].update({f: text_idx[f].get(word)})
            else:
                master_idx[word] = {f: text_idx[f].get(word)}  # mappa word i master_idx till en ny lista
                                                               # varje gång vid nytt ord, vars nyckel är namnet
    return master_idx                                          # på boken och värdet är listan av förekomster


def concordance(word, master_index, window):
    for w in master_index:  # för varje nyckel w in master_index
        if w == word:
            dict_txt_file = master_index[w]
            for txt_file in dict_txt_file:  # för varje nyckel w in dict_txt_file
                file = open('Selma/' + txt_file, encoding='utf-8')
                file_text = file.read()

                print(txt_file)
                list_of_index = dict_txt_file[txt_file]
                for idx in list_of_index:
                    row = '        ' + file_text[max(idx - window, 0):min(idx + window, len(file_text) - 1)]
                    row = row.replace('\n', ' ')
                    print(row)


def tfidf(master_index, files):                             # FUNKAR INTE!!!!! NÅGOT LITET ÄR FEL
    nbr_of_files = len(files)
    total_nbr_word = {}

    for file_name in files:  # initiera för att vi ej ska addera ett tomt nyckel-vÄrde med en int
        total_nbr_word[file_name] = 0

    for word in master_index:  # mappar alla filnamn till totala antal ord i respektive fil
        file_name_dict = master_index[word]

        for file_name in file_name_dict:
            pos_list = file_name_dict[file_name]
            nbr_word_occ = len(pos_list)
            total_nbr_word[file_name] = nbr_word_occ + total_nbr_word[file_name]

    tf_idf = {}
    word_tfidf = {}
    for word in master_index:  # mappar alla filnamn till totala antal ord i respektive fil
        file_name_dict = master_index[word]

        for file_name in file_name_dict:
            pos_list = file_name_dict[file_name]
            nbr_word_occ = len(pos_list)
            tf = nbr_word_occ / total_nbr_word[file_name]
            if (tf != 0):
                idf = math.log10(nbr_of_files / len(file_name_dict))
            else:
                idf = 0

            word_tfidf[word] = tf * idf
            tf_idf[file_name] = word_tfidf

    return tf_idf


"""def tfidf(master_index, files): OBS DENNA VERSIONEN FRÅN AXEL FUNKAR, ANVÄNDA DENNA OM DU VILL KÖRA PRORAMMET
    tf = {}
    idf = {}
    tf_idf = {}
    for file in files:
        text = open('Selma/' + file, encoding = 'utf-8').read().lower().strip()
        total_nbr_word = len(re.findall(regex, text))
        tf[file] = {}
        tf_idf[file] = {}
        for w in master_index:
            if master_index[w].get(file) is None:
                freq = 0.0
            else:
                freq = len(master_index[w].get(file))
            tf[file].update({w: freq/total_nbr_word})
    N = len(files)
    for w in master_index:
        idf[w] = math.log(N / float((len(master_index[w].keys()))), 10)
    for file in files:
        for w in tf[file]:
            tf_idf[file][w] = tf[file].get(w) * idf[w]
    return tf_idf"""


def cosine_similarity(doc_1, doc_2, tf_idf):  # FUNKAR INTE!!!!!
    a = np.array([])
    b = np.array([])
    for word in tf_idf[doc_1]:  # funkar med tf_idf[doc_2] också eftersom de innehåller samma (alla) ord
        np.append(a, tf_idf[doc_1][word])
        np.append(b, tf_idf[doc_2][word])

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    return dot_product / (norm_a * norm_b)

"""def cosine_similarity(doc_1, doc_2, tf_idf): # OBS DENNA VERSIONEN FRÅN AXEL FUNKAR, ANVÄNDA DENNA OM DU VILL KÖRA PRORAMMET
    
    a = tf_idf[doc_1]
    b = tf_idf[doc_2]
    
    sum_tf_idf = 0
    pow_a = 0
    pow_b = 0
    for word in tf_idf[doc_1]: # funkar med tf_idf[doc_2] också eftersom de innehåller samma (alla) ord
        sum_tf_idf = sum_tf_idf + a[word]*b[word]
        pow_a = pow_a + pow(a[word], 2)
        pow_b = pow_b + pow(b[word], 2)
    
    similarity = sum_tf_idf/(math.sqrt(pow_a)*math.sqrt(pow_b))

    return similarity"""


if __name__ == '__main__':
    print('indexerLab1')

    """ 
    #  exempel på hur man sparar lite data i en pickle-file
    file = open('Selma/marbacka.txt', encoding='utf-8')  # måste ha utf-8 annars kan den ej läsa in vissa filer
    file_text = file.read() # läser in filen (lowercase sköts av tokenize) 
    tokens = tokenize(file_text) # file_text är en string, inte en lista
    idx = text_to_idx(tokens) # tokens är en lista av ord
    pickle.dump(idx, open("idx.p", "wb"))  # .p betyder att det är en pickle-fil, w = writing, b = binary (onödig info)
    idx = pickle.load(open("idx.p", "rb"))  # load the dictionary back from the pickle file, r = reading
    print(files)

    my_master_index = master_index(files)
    my_master_index['samlar']"""

    pattern = r'\p{L}+'  # unicode commando for finding sequences of letters, i.e. words
    re.findall(pattern, 'En gång hade de på Mårbacka en barnpiga, som hette Back-Kajsa')

    files = get_files('Selma', 'txt')
    size = len(files)
    limit = -1
    max_similarity = 0
    most_sim_doc1 = ''
    most_sim_doc2 = ''
    print(' '*len(files[0]) + ' '.join(files).replace('.txt', ''))

    for row in range(0, size):
        list = []
        for col in range(0, size):
            sim = 3  # should be (when tfidf is working): round(cosine_similarity(files[row], files[col], my_tf_idf), 5)
            list.append(str(sim))
            if ((sim != 1) and (sim > limit)):
                limit = sim
                max_similarity = sim
                most_sim_doc1 = files[row]
                most_sim_doc2 = files[col]

        indent = 20 - len(files[row])
        print(files[row].replace('.txt', '') + ' '*indent + ' '.join(list))
        my_master_index = master_index(files)

    print(my_master_index['kukeliku'])

    print(max(1,2))