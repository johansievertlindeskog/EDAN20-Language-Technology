
import os
import regex as re


def read_sentences(file):
    """
    Creates a list of sentences from the corpus
    Each sentence is a string
    :param file:
    :return:
    """
    f = open(file, encoding='utf-8').read().strip()
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
    root_values = ['0', 'ROOT', 'ROOT', 'ROOT', 'ROOT', 'ROOT', '0', 'ROOT', '0', 'ROOT']
    start = [dict(zip(column_names, root_values))]
    for sentence in sentences:
        rows = sentence.split('\n')
        sentence = [dict(zip(column_names, row.split('\t'))) for row in rows if row[0] != '#']
        sentence = start + sentence
        new_sentences.append(sentence)
    return new_sentences


def convert_to_dict(formatted_corpus):
    """
    Converts each sentence from a list of words to a dictionary where the keys are id
    :param formatted_corpus:
    :return:
    """
    formatted_corpus_dict = []
    for sentence in formatted_corpus:
        sentence_dict = {}
        for word in sentence:
            sentence_dict[word['ID']] = word
        formatted_corpus_dict.append(sentence_dict)
    return formatted_corpus_dict


def extract_pairs(formatted_corpus_dict):
    pair_count_dict = {}
    for sentence in formatted_corpus_dict:
        for word_idx in sentence:
            if sentence[word_idx]['DEPREL'].startswith('nsubj'): # varje träff av subjekt
                    pair = []
                    pair.append(sentence[word_idx]['FORM'].lower()) # lägger till subjektet
                    pair.append(sentence[sentence[word_idx]['HEAD']]['FORM'].lower())
                    tp = tuple(pair)
                    if tp in pair_count_dict:
                        pair_count_dict[tp] = pair_count_dict[tp] + 1
                    else:
                        pair_count_dict[tp] = 1

    return pair_count_dict


def extract_triples(formatted_corpus_dict):
    triples_count_dict = {}
    for sentence in formatted_corpus_dict:
        for word_idx in sentence:
            if sentence[word_idx]['DEPREL'].startswith('nsubj'):  # varje träff av subjekt
                triples_list = []
                triples_list.append(sentence[word_idx]['FORM'].lower())  # lägger till subjektet
                verb_idx = sentence[word_idx]['HEAD']
                triples_list.append(sentence[verb_idx]['FORM'].lower())  # lägger till verbet
                for word_idx in sentence:  # när vi har träff för subjekt-verb måste vi leta efter objekt med verb_idx
                    if sentence[word_idx]['DEPREL'].startswith('obj') and verb_idx == sentence[word_idx][
                        'HEAD']:  # måste ha == för annars kan vi få träff på HEAD:12 när vi söker efter HEAD:1
                        triples_list.append(sentence[word_idx]['FORM'].lower())  # lägger till objektet
                        tp = tuple(
                            triples_list)  # notera att man kan göra tuples på ett annat sätt också, se extract_entity_triples nedan
                        if tp in triples_count_dict:
                            triples_count_dict[tp] = triples_count_dict[tp] + 1
                        else:
                            triples_count_dict[tp] = 1

    return triples_count_dict


def extract_pairs_and_triples(formatted_corpus_dict, nbest):
    pairs = extract_pairs(formatted_corpus_dict)
    sorted_pairs = sorted(pairs, key=lambda x: (-pairs[x], x))
    frequent_pairs = [(pair, pairs[pair]) for pair in sorted_pairs][:nbest]

    triples = extract_triples(formatted_corpus_dict)
    sorted_triples = sorted(triples, key=lambda x: (-triples[x], x))
    frequent_triples = [(triple, triples[triple]) for triple in sorted_triples][:nbest]

    return frequent_pairs, frequent_triples


def extract_entity_triples(formatted_corpus_dict):
    triples = []
    for sentence in formatted_corpus_dict:
        for word_idx in sentence:
            if sentence[word_idx]['DEPREL'].startswith('nsubj') and sentence[word_idx]['UPOS'].startswith(
                    'PROPN'):  # varje träff av subjekt
                subj = sentence[word_idx]['FORM']  # sparar subjektet
                verb_idx = sentence[word_idx]['HEAD']
                verb = sentence[verb_idx]['FORM'].lower()  # sparar verbet
                for word_idx in sentence:  # när vi har träff för subjekt-verb måste vi leta efter objekt med verb_idx
                    if sentence[word_idx]['DEPREL'].startswith('obj') and verb_idx == sentence[word_idx]['HEAD'] and \
                            sentence[word_idx]['UPOS'].startswith(
                                    'PROPN'):  # måste ha == för annars kan vi få träff på HEAD:12 när vi söker efter HEAD:1
                        obj = sentence[word_idx]['FORM']  # sparar objektet
                        subj_verb_obj = (subj, verb, obj)
                        if subj_verb_obj not in triples:
                            triples.append(subj_verb_obj)

    return triples


def get_files(dir, suffix):
    """
    Returns all the files in a folder ending with suffix
    Recursive version
    :param dir:
    :param suffix:
    :return: the list of file names
    """
    files = []
    for file in os.listdir(dir):
        path = dir + '/' + file
        if os.path.isdir(path):
            files += get_files(path, suffix)
        elif os.path.isfile(path) and file.endswith(suffix):
            files.append(path)
    return files


if __name__ == '__main__':
    print('extractionOfTriplesLab5')

    ud_path = 'corpus/ud-treebanks-v2.6/'
    path_sv = ud_path + 'UD_Swedish-Talbanken/sv_talbanken-ud-train.conllu'
    path_fr = ud_path + 'UD_French-GSD/fr_gsd-ud-train.conllu'
    path_ru = ud_path + 'UD_Russian-SynTagRus/ru_syntagrus-ud-train.conllu'
    path_en = ud_path + 'UD_English-EWT/en_ewt-ud-train.conllu'
    column_names_u = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']

    sentences = read_sentences(path_sv)
    formatted_corpus = split_rows(sentences, column_names_u)
    print(len(formatted_corpus))  # ska bli 4303

    # convert the formatted_corpus to dict
    formatted_corpus_dict = convert_to_dict(formatted_corpus)

    # Extracting the subject-verb pairs
    pairs_sv = extract_pairs(formatted_corpus_dict)
    print(sum([pairs_sv[pair] for pair in pairs_sv]))  # ska bli 6083

    # Finding the most frequent pairs
    nbest = 3
    sorted_pairs = sorted(pairs_sv, key=lambda x: (-pairs_sv[x], x))
    freq_pairs_sv = [(pair, pairs_sv[pair]) for pair in sorted_pairs][:nbest]
    print(freq_pairs_sv)

    # Extracting the subject-verb-object triples
    triples_sv = extract_triples(formatted_corpus_dict)
    print(sum([triples_sv[triple] for triple in triples_sv]))  # ska bli 2054

    # Finding the most frequent triples
    sorted_triples = sorted(triples_sv, key=lambda x: (-triples_sv[x], x))
    freq_triples_sv = [(triple, triples_sv[triple]) for triple in sorted_triples][:nbest]
    print(freq_triples_sv)

    # repeating for all the languages
    files = get_files(ud_path, 'train.conllu')
    list_frequent_pairs = []
    list_frequent_triples = []
    for file in files:
        sentences = read_sentences(file)
        formatted_corpus = split_rows(sentences, column_names_u)
        formatted_corpus_dict = convert_to_dict(formatted_corpus)
        freq_pair, freq_triple = extract_pairs_and_triples(formatted_corpus_dict, nbest)

        list_frequent_pairs.append(freq_pair)
        list_frequent_triples.append(freq_triple)

    #  extract nbest pairs and triples in French, Russian, and English
    sentences = read_sentences(path_fr)
    formatted_corpus = split_rows(sentences, column_names_u)
    formatted_corpus_dict = convert_to_dict(formatted_corpus)
    freq_pairs_fr, freq_triples_fr = extract_pairs_and_triples(formatted_corpus_dict, nbest)
    print(freq_pairs_fr)
    print(freq_triples_fr)

    sentences = read_sentences(path_ru)
    formatted_corpus = split_rows(sentences, column_names_u)
    formatted_corpus_dict = convert_to_dict(formatted_corpus)
    freq_pairs_ru, freq_triples_ru = extract_pairs_and_triples(formatted_corpus_dict, nbest)
    print(freq_pairs_ru)
    print(freq_triples_ru)

    sentences = read_sentences(path_en)
    formatted_corpus = split_rows(sentences, column_names_u)
    formatted_corpus_dict = convert_to_dict(formatted_corpus)
    freq_pairs_en, freq_triples_en = extract_pairs_and_triples(formatted_corpus_dict, nbest)
    print(freq_pairs_en)
    print(freq_triples_en)

    # Resolving the entities
    nbest = 5
    entity_relation_en = extract_entity_triples(formatted_corpus_dict)
    entity_relation_en = sorted(entity_relation_en)[:nbest]
    print(entity_relation_en)


