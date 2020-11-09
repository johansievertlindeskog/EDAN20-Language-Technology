import math
import regex as re
from scipy import stats


def tokenize(text):
    words = re.findall(r'\p{L}+', text)
    return words


def clean(text):
    nonletter = r'[^.;:?!\p{L}]'
    cleanText = re.sub(nonletter, ' ', text)
    cleanText = cleanText.replace('  ', ' ')
    return cleanText


def substitution(sentence_boundaries, sentence_markup, test_para):
    text = re.sub(sentence_boundaries, sentence_markup, test_para)
    return text


def segment_sentences(text):
    sentence_boundaries = r'([.:;!?])(\p{Z}+)(\p{Lu})'  # paranteser runt varje tecken för att matcha kombinationen
    sentence_markup = r' </s>\n<s> \3'          # av alla 3 men ha referens till subgrupperna. 3an är referensen till
    text1 = substitution(sentence_boundaries, sentence_markup, text) # den stora bokstaven från sentence_boundaries
    text1 = '<s> ' + text1 + ' </s>'
    text1 = re.sub(r'\p{Z}+', ' ', text1)
    text1 = re.sub(r'[.:;!?]', '', text1)
    return text1.lower()


def unigrams(words):
    frequency = {}
    for i in range(len(words)):
        if words[i] in frequency:
            frequency[words[i]] += 1
        else:
            frequency[words[i]] = 1
    return frequency


def bigrams(words):
    bigrams = []
    for i in range(len(words) - 1):
        bigrams.append((words[i], words[i + 1]))
    frequency_bigrams = {}
    for i in range(len(words) - 1):
        if bigrams[i] in frequency_bigrams:
            frequency_bigrams[bigrams[i]] += 1
        else:
            frequency_bigrams[bigrams[i]] = 1
    return frequency_bigrams


def trigrams(words):
    trigrams = []
    for i in range(len(words) - 2):
        trigrams.append((words[i], words[i + 1], words[i + 2]))
    frequency_trigrams = {}
    for i in range(len(words) - 2):
        if trigrams[i] in frequency_trigrams:
            frequency_trigrams[trigrams[i]] += 1
        else:
            frequency_trigrams[trigrams[i]] = 1
    return frequency_trigrams


def sentence_prob_unigram(sentence, frequency_unigrams):
    tot_nbr_words_corpus = 0
    for word in frequency_unigrams:
        tot_nbr_words_corpus = tot_nbr_words_corpus + frequency_unigrams[word]

    tot_nbr_words_corpus = tot_nbr_words_corpus - 1 # får ett ord för mycket
    cleaned_sentence = clean(sentence)
    segmented_sentence = segment_sentences(cleaned_sentence)
    words_in_sentence = segmented_sentence.split()  # carriage return = \n och \r

    word_prob = {}
    unigram_prob_list = []
    sentence_prob = 1
    backoff = 8.352285982272032e-05
    for word in words_in_sentence:
        if word in frequency_unigrams:
            word_prob[word] = frequency_unigrams[word] / tot_nbr_words_corpus
        else:
            word_prob[word] = backoff

        unigram_prob_list.append(word_prob[word])
        sentence_prob = sentence_prob * word_prob[word]

    geometric_mean_prob = stats.gmean(unigram_prob_list)
    entropy_rate = -1*math.log2(sentence_prob)/len(words_in_sentence)
    perplexity = 2 ** entropy_rate

    # For the print
    line = "====================================================="
    print(line)
    print("wi", '\t'*2, "C(wi)", '\t', "#words", '\t', "P(wi)")
    print(line)
    i = 0
    for word in words_in_sentence:
        tab = '\t'  # oss lägger till lite random kodning för att det ska bli snyggt
        if len(words_in_sentence[i]) < 3:
            tab = 2*tab

        if word in frequency_unigrams:
            print(words_in_sentence[i], tab, frequency_unigrams[word], '\t', tot_nbr_words_corpus, '\t', word_prob[word])
        else:
            print(words_in_sentence[i], tab, 0, '\t', tot_nbr_words_corpus, '\t', 0, '*Backoff:', backoff)
        i = i + 1

    print(line)
    print('Prob. unigrams:', '\t'*3, sentence_prob, '\n' + 'Geometric mean prob.:', '\t'*2, geometric_mean_prob, '\n' + 'Entropy rate:', '\t'*4, entropy_rate, '\n' + 'Perplexity:', '\t'*4, perplexity)

    return perplexity


def sentence_prob_bigram(sentence, frequency_unigrams, frequency_bigrams):
    cleaned_sentence = clean(sentence)
    segmented_sentence = segment_sentences(cleaned_sentence)
    words_in_sentence = segmented_sentence.split()  # carriage return = \n och \r

    bigrams_in_sentence = {}
    for i in range(len(words_in_sentence)):
        if i + 1 < len(words_in_sentence):
            bigram_to_tuple = tuple([words_in_sentence[i], words_in_sentence[i + 1]])  # tuple kräver 1 objekt
            if bigram_to_tuple in frequency_bigrams:
                bigrams_in_sentence[bigram_to_tuple] = frequency_bigrams[bigram_to_tuple]
            else:
                bigrams_in_sentence[bigram_to_tuple] = 0

    bigram_prob = {}
    sentence_prob = 1
    bigram_prob_list = []
    backoff = 8.352285982272032e-05
    for tp in bigrams_in_sentence:
        if tp in frequency_bigrams:
            bigram_prob[tp] = frequency_bigrams[tp] / frequency_unigrams[tp[0]]
        else:
            bigram_prob[tp] = 0

        if bigram_prob[tp] != 0:
            sentence_prob = sentence_prob * bigram_prob[tp]
            bigram_prob_list.append(bigram_prob[tp])
        else:
            sentence_prob = sentence_prob * backoff
            bigram_prob_list.append(backoff)

    geometric_mean_prob = stats.gmean(bigram_prob_list)
    entropy_rate = -1 * math.log2(sentence_prob) / len(bigrams_in_sentence)
    perplexity = 2 ** entropy_rate

    # For the print
    line = "====================================================="
    print(line)
    print("wi", '\t' * 2, "wi+1", '\t', "Ci,i+1", '\t', "Ci", '\t', "P(wi+1|wi)")
    print(line)

    for tp in bigrams_in_sentence:
        if tp in frequency_bigrams:
            print(tp[0], '\t', tp[1], '\t', frequency_bigrams[tp], '\t', frequency_unigrams[tp[0]], '\t',
                  bigram_prob[tp])
        elif tp[0] in frequency_unigrams:
            print(tp[0], '\t', tp[1], '\t', 0, '\t', frequency_unigrams[tp[0]], '\t', bigram_prob[tp], '*Backoff:',
                  backoff)
        else:
            print(tp[0], '\t', tp[1], '\t', 0, '\t', 0, '\t', bigram_prob[tp], '*Backoff:', backoff)

    print(line)
    print('Prob. bigrams:', '\t' * 4, sentence_prob, '\n' + 'Geometric mean prob.:', '\t' * 2, geometric_mean_prob,
          '\n' + 'Entropy rate:', '\t' * 4, entropy_rate, '\n' + 'Perplexity:', '\t' * 4, perplexity)

    return perplexity


def prediction1(starting_text, frequency_bigrams, cand_nbr):
    current_word_predictions_1 = []
    candidates_prediction = {}
    starting_text = starting_text.lower()
    for bigram in frequency_bigrams:
        s1 = bigram[0]
        s2 = bigram[1]
        match = re.search(starting_text, s2)
        pos = -1
        if match is not None:
            pos = match.start()
        if (s1 == '<s>') & (pos == 0):
            candidates_prediction[s2] = frequency_bigrams.get(bigram)

    candidates_prediction_sorted = sorted(candidates_prediction.items(), key=lambda x: x[1], reverse=True)
    for candidate in candidates_prediction_sorted:
        current_word_predictions_1.append(candidate[0])  # notera att "sorted" ovan ger mig en lista och att jag därför
                                                         # behöver plocka ut candidate[0] istället för bara candidate
    limit = len(current_word_predictions_1)  # python tar egentligen hand om en eventuell kortare lista än cand_nbr
    if limit > cand_nbr:                     # så hela denna sista checken är onödig
        limit = cand_nbr

    return current_word_predictions_1[0:limit]


def prediction_next_word(starting_text, frequency_trigrams, cand_nbr):
    next_word_predictions = []
    candidates_prediction = {}
    starting_text_lowered = starting_text.lower()
    list_starting_text_lowered = starting_text_lowered.split()
    tuple_starting_text = tuple(list_starting_text_lowered)

    pos = len(list_starting_text_lowered) - 2
    pos_last = len(list_starting_text_lowered) - 1
    for trigram in frequency_trigrams:
        s0 = trigram[0]
        s1 = trigram[1]
        if (tuple_starting_text[pos] == s0) and (tuple_starting_text[pos_last] == s1):
            candidates_prediction[trigram] = frequency_trigrams[trigram]

    candidates_prediction_sorted = sorted(candidates_prediction.items(), key=lambda x: x[1], reverse=True)
    for candidate in candidates_prediction_sorted:
        next_word_predictions.append(candidate[0][2])

    return next_word_predictions[0:cand_nbr]


def prediction2(starting_text, frequency_trigrams, cand_nbr):
    current_word_predictions_2 = []
    candidates_prediction = {}
    starting_text_lowered = starting_text.lower()
    list_starting_text_lowered = starting_text_lowered.split()
    tuple_starting_text = tuple(list_starting_text_lowered)

    pos = len(list_starting_text_lowered) - 3
    pos_last = len(list_starting_text_lowered) - 2
    for trigram in frequency_trigrams:
        s0 = trigram[0]
        s1 = trigram[1]
        if (tuple_starting_text[pos] == s0) and (tuple_starting_text[pos_last] == s1):
            s2 = trigram[2]
            if tuple_starting_text[3] == s2[0]:
                candidates_prediction[trigram] = frequency_trigrams[trigram]

    candidates_prediction_sorted = sorted(candidates_prediction.items(), key=lambda x: x[1], reverse=True)
    for candidate in candidates_prediction_sorted:
        current_word_predictions_2.append(candidate[0][2])

    return current_word_predictions_2[0:cand_nbr]


if __name__ == '__main__':
    print('languageModelsLab2')
    corpus = open('Selma.txt', encoding='utf-8').read()
    cleaned_corpus = clean(corpus)
    segmented_corpus = segment_sentences(cleaned_corpus)
    words_corpus = re.split(r' |\n|\r', segmented_corpus)  # carriage return = \n och \r
    frequency_unigrams = unigrams(words_corpus)
    frequency_bigrams = bigrams(words_corpus)
    frequency_trigrams = trigrams(words_corpus)

    current_word_predictions_1 = prediction1('De', frequency_bigrams, 5)
    next_word_predictions = prediction_next_word("Det var en ", frequency_trigrams, 5)
    current_word_predictions_2 = prediction2("Det var en g", frequency_trigrams, 5)
    print(current_word_predictions_1[0:5])
    print(next_word_predictions[0:5])
    print(current_word_predictions_2[0:5])

    perplexity1 = sentence_prob_unigram('Detta var en gång en korv som hette Nils.', frequency_unigrams)
    perplexity2 = sentence_prob_bigram('Sho katt, vad gör du här?', frequency_unigrams, frequency_bigrams)
