import sys
import time
import numpy as np
from collections import defaultdict

STUDENT = {'name': 'Alex Kartun_Ofir Sharon',
           'ID': '324429216_204717664'}


START_SYMBOL = 'start'
NUM_WORDS_SYMBOL = 'NUM_WORDS'
EXTRA_FILE_NAME = 'ass1-tagger-extra'
RARE_WORD_MAX_FREQ = 1


def extract_raw_data(train_fname):
    """
    Extract tags and words list from training corpus.
    :param train_fname: File name of training file.
    :return: List of tags and words respectively.
    """
    with open(train_fname, 'r') as f:
        lines = f.readlines()

    train_raw_data = []
    for line in lines:
        words = []
        tags = []
        line_parts = line.strip().split()
        for part in line_parts:
            word, tag = part.rsplit('/', 1)
            words.append(word)
            tags.append(tag)
        train_raw_data.append((words, tags))

    return train_raw_data


def get_counters(data):
    """
    Calculate emission and transaction counts from the corpus.
    :param data: Data containing lists of words and tags.
    :return: Mapping of emiss and trans to counts.
    """
    emission_c = defaultdict(int)
    transaction_c = defaultdict(int)

    for words, tags in data:
        # Calculate emission counts and number of words count.
        for ind, word in enumerate(words):
            transaction_c[NUM_WORDS_SYMBOL] += 1
            key = '{} {}'.format(word, tags[ind])
            emission_c[key] += 1
        # Calculate transaction counts from the tags.
        for tag_1, tag_2, tag_3 in zip(tags[0:], tags[1:], tags[2:]):
            key = tag_1
            transaction_c[key] += 1
            key = '{} {}'.format(tag_1, tag_2)
            transaction_c[key] += 1
            key = '{} {} {}'.format(tag_1, tag_2, tag_3)
            transaction_c[key] += 1
        # Add additional transactions that beginning with start symbol.
        if len(tags) > 0:
            key = '{} {} {}'.format(START_SYMBOL, START_SYMBOL, tags[0])
            transaction_c[key] += 1

        if len(tags) > 1:
            key = '{} {} {}'.format(START_SYMBOL, tags[0], tags[1])
            transaction_c[key] += 1

        key = '{} {}'.format(START_SYMBOL, START_SYMBOL)
        transaction_c[key] += 1
        key = START_SYMBOL
        transaction_c[key] += 1
    # Extract rare words whose count equal to assign value in RARE_WORD_MAX_FREQ.
    rare_words_and_tags = [key.split() for key, value in emission_c.items() if value == RARE_WORD_MAX_FREQ]
    # For each rare word convert it so his signature and insert new emission to the map.
    for word, tag in rare_words_and_tags:
        signature_word = subcategorize(word)
        key = '{} {}'.format(signature_word, tag)
        emission_c[key] += 1
    # Calculate deleted interpolation lambdas for smoothing.
    interp_weights = calculate_deleted_interpolation(transaction_c)

    return emission_c, transaction_c, interp_weights


def subcategorize(word):
    """
    Convert word to his signature word.
    :param word:
    :return:
    """
    # Contains digit.
    if word.isdigit():
        return '^NUM'
    # Contains hyphen.
    if '-' in word:
        return '^HYPHEN'
    # Contains not alphabetic symbols.
    if not word.isalpha():
        return '^$'
    # Starting with upper case letter and remaining are lower and ending with one of those suffixes.
    if word.istitle():
        if word.endswith('ing'):
            return '^X_ing'
        if word.endswith('ed'):
            return '^X_ed'
        if word.endswith('tial'):
            return '^X_tial'
        if word.endswith('s'):
            return '^X_s'
        if word.endswith('ly'):
            return '^X_ly'
        if word.endswith('er'):
            return '^X_er'
    # Not start with upper case letter but ending with one of those suffixes.
    else:
        if word.endswith('ing'):
            return '^_ing'
        if word.endswith('ed'):
            return '^_ed'
        if word.endswith('tial'):
            return '^_tial'
        if word.endswith('s'):
            return '^_s'
        if word.endswith('ly'):
            return '^_ly'
        if word.endswith('er'):
            return '^_er'
    # The word is starting with upper case letter all remaining are lower case.
    if word.istitle():
        return '^Aa'
    # All the letters are upper case.
    if word.isupper():
        return '^AA'
    # Otherwise return unk symbol.
    return '^UNK'


def output(q_mle_fname, e_mle_fname, transaction_c, emission_c, interp_weights):
    """
    Write the counts and interpolation lambdas to the files.
    :param interp_weights: Interpolation weights. Write them to extra file.
    :param q_mle_fname: Q mle file name.
    :param e_mle_fname: E mle file name.
    :param transaction_c: Transaction counts.
    :param emission_c: Emission counts.
    :return:
    """
    trans_output_data = ['{}\t{}'.format(key, value) for key, value in transaction_c.items()]
    with open(q_mle_fname, 'w') as q_file:
        q_file.write('\n'.join(trans_output_data))

    emission_output_data = ['{}\t{}'.format(key, value) for key, value in emission_c.items()]
    with open(e_mle_fname, 'w') as e_file:
        e_file.write('\n'.join(emission_output_data))

    with open(EXTRA_FILE_NAME, 'w') as extra_file:
        extra_file.write(' '.join([str(v) for v in interp_weights]))


def train(train_fname, q_mle_fname, e_mle_fname):
    """
    Train the training corpus. By extracting the data and calculate counts and lambdas.
    :param train_fname: Train file name.
    :param q_mle_fname: Q file name.
    :param e_mle_fname: E file name.
    :return:
    """
    start = time.time()

    print 'starting to extract raw data...'
    train_raw_data = extract_raw_data(train_fname)
    print 'creating emission and transaction counts...'
    emission_c, transaction_c, interp_weights = get_counters(train_raw_data)
    print 'writing the counts to the files...'
    output(q_mle_fname, e_mle_fname, transaction_c, emission_c, interp_weights)

    end = time.time()
    print('time took to train: ' + str(end - start) + ' sec')


def load_model(q_mle_fname, e_mle_fname, extra_file_name):
    """
    Load the model from the files. Extract all the counts and lambdas from the files.
    :param q_mle_fname: Q mle file name.
    :param e_mle_fname: E mle file name.
    :param extra_file_name: Extra file name.
    :return: All the extracted data.
    """
    transaction_c = defaultdict(int)
    emission_c = defaultdict(int)
    tag_set = set()
    word_to_tags_map = defaultdict(set)

    with open(q_mle_fname, 'r') as q_file:
        q_data = q_file.readlines()
    with open(e_mle_fname, 'r') as e_file:
        e_data = e_file.readlines()
    with open(extra_file_name, 'r') as extra_file:
        extra_data = extra_file.readlines()
    # Extract transaction counts and create tag set.
    for line in q_data:
        key, val = line.strip().split('\t')
        transaction_c[key] = val
        for tag in key.split():
            tag_set.add(tag)
    # Extract emission counts and create word to tags map. Which mapping from the word to list of his tags.
    for line in e_data:
        key, val = line.strip().split('\t')
        emission_c[key] = val
        word, tag = key.split()
        word_to_tags_map[word].add(tag)
    # Extract interpolation weights.
    interpolation_weights = [float(v) for data in extra_data for v in data.strip().split()]

    return transaction_c, emission_c, tag_set, word_to_tags_map, interpolation_weights


def get_e(word, tag, emiss_c, trans_c):
    """
    Calculate e probability.
    :param word: Current word to analyze.
    :param tag: Current tag to analyze.
    :param emiss_c: Emission counts.
    :param trans_c: Transaction counts.
    :return: Probability value.
    """
    key = '{} {}'.format(word, tag)
    count_word_tag = float(emiss_c.get(key, 0))
    key = '{}'.format(tag)
    count_tag = float(trans_c.get(key, 0))
    try:
        e_prob = count_word_tag / count_tag
    except ZeroDivisionError:
        e_prob = 0

    return e_prob


def get_q(pp_t, p_t, curr_tag, trans_c, interpolation_weights):
    """
    Calculate q probability.
    :param pp_t: Previous of previous tag to analyze.
    :param p_t: Previous tag to analyze.
    :param curr_tag: Current tag to analyze.
    :param trans_c: Transaction counts.
    :param interpolation_weights: Interpolation weight.
    :return:
    """
    lambda_1 = interpolation_weights[0]
    lambda_2 = interpolation_weights[1]
    lambda_3 = interpolation_weights[2]
    # Calculate trigram prob = count_trigram_abc / count_bigram_ab
    key = '{} {} {}'.format(pp_t, p_t, curr_tag)
    count_trigram_abc = float(trans_c.get(key, 0))
    key = '{} {}'.format(pp_t, p_t)
    count_bigram_ab = float(trans_c.get(key, 0))
    try:
        trigram_prob = count_trigram_abc / count_bigram_ab
    except ZeroDivisionError:
        trigram_prob = 0
    # Calculate bigram prob = count_trigram_bc / count_unigram_b
    key = '{} {}'.format(p_t, curr_tag)
    count_bigram_bc = float(trans_c.get(key, 0))
    key = '{}'.format(p_t)
    count_unigram_b = float(trans_c.get(key, 0))
    try:
        bigram_prob = count_bigram_bc / count_unigram_b
    except ZeroDivisionError:
        bigram_prob = 0
    # Calculate unigram prob = count_unigram_c / num_words
    key = '{}'.format(curr_tag)
    count_unigram_c = float(trans_c.get(key, 0))
    key = '{}'.format(NUM_WORDS_SYMBOL)
    num_words = float(trans_c.get(key, 0))
    try:
        unigram_prob = count_unigram_c / num_words
    except ZeroDivisionError:
        unigram_prob = 0
    # Apply interpolation weight on the probabilities.
    interpolation = lambda_1 * trigram_prob + lambda_2 * bigram_prob + lambda_3 * unigram_prob
    return interpolation


def calculate_deleted_interpolation(trans_c):
    """
    Calculate linear deleted interpolation lambdas for smoothing the values.
    :param trans_c: Transaction counts.
    :return: Deleted interpolation lambdas.
    """
    lambda1 = lambda2 = lambda3 = 0
    sum_of_all_unigrams_counts = sum(v for k, v in trans_c.items() if len(k.split()) == 1)
    sum_of_all_unigrams_counts -= (trans_c.get(NUM_WORDS_SYMBOL, 0))
    for key, value in trans_c.items():
        if len(key.split()) != 3:
            continue
        a, b, c = key.split()
        if value > 0:
            try:
                key = '{} {}'.format(a, b)
                c1 = float(value - 1) / (trans_c.get(key, 1) - 1)
            except ZeroDivisionError:
                c1 = 0
            try:
                key = '{} {}'.format(a, b)
                c2 = float(trans_c.get(key, 1) - 1) / (trans_c.get(a, 1) - 1)
            except ZeroDivisionError:
                c2 = 0
            try:
                c3 = float(trans_c.get(a, 1) - 1) / (sum_of_all_unigrams_counts - 1)
            except ZeroDivisionError:
                c3 = 0

            k = np.argmax([c1, c2, c3])
            if k == 0:
                lambda1 += value
            if k == 1:
                lambda2 += value
            if k == 2:
                lambda3 += value

    weights = [lambda1, lambda2, lambda3]
    norm_w = [float(a) / sum(weights) for a in weights]
    return norm_w


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print 'need at least 3 arguments.'
        exit(1)

    train_file_name = sys.argv[1]
    q_mle_file_name = sys.argv[2]
    e_mle_file_name = sys.argv[3]

    train(train_file_name, q_mle_file_name, e_mle_file_name)
