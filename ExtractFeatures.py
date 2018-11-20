import sys
import time
import re
from collections import defaultdict

STUDENT = {'name': 'Alex Kartun_Ofir Sharon',
           'ID': '324429216_204717664'}


START_SYMBOL = 'start'
RARE_WORD_MAX_FREQ = 1


def extract_raw_data(train_fname):
    """
    Extract list of words and tags from the train corpus. And create map of words to their counts.
    :param train_fname: Train file name.
    :return: Lists of words and tags and words map.
    """
    with open(train_fname, 'r') as f:
        lines = f.readlines()

    train_raw_data = []
    word_counters = defaultdict(int)
    for line in lines:
        words = []
        tags = []
        line_parts = line.strip().split()
        for part in line_parts:
            word, tag = part.rsplit('/', 1)
            words.append(word)
            tags.append(tag)
            word_counters[word] += 1
        train_raw_data.append((words, tags))

    return train_raw_data, word_counters


def output(features_fname, all_corpus_features):
    """
    Output all list of features to the features file.
    :param features_fname: Features file name.
    :param all_corpus_features: List of all features.
    :return:
    """
    with open(features_fname, 'w') as f_file:
        f_file.write('\n'.join(all_corpus_features))


def generate_word_features(is_rare, p_t, pp_t, curr_word, word_ind, words):
    """
    Generate for current word list of features as listed on table one.
    :param is_rare: Boolean that corresponds to type of the word. Rare or not.
    :param p_t: Previous tag.
    :param pp_t: Previous of previous tag.
    :param curr_word: Current word.
    :param word_ind: Current word index.
    :param words: List of all words in the sentence.
    :return: List of all word features.
    """
    word_features = []
    # Check if word is rare.
    if is_rare:
        # Generate the suffixes and prefixes depends on min of (word length or 4).
        for i in range(1, min(5, len(curr_word))):
            word_features.append('prefix' + str(i) + '=' + curr_word[:i])
        for i in range(1, min(5, len(curr_word))):
            word_features.append('suffix' + str(i) + '=' + curr_word[-i:])
        # Check with regex if word contains digit.
        if re.search(r'\d', curr_word):
            word_features.append('has_digit=true')
        # Check with regex if word contains upper case letter.
        if re.search(r'[A-Z]', curr_word):
            word_features.append('has_upper_letter=true')
        # Check if word contains hyphen symbol.
        if '-' in curr_word:
            word_features.append('has_hyphen=true')
    else:
        # Otherwise word is not rare and this word as feature.
        key = 'form={}'.format(curr_word)
        word_features.append(key)
    # Generate previous tags.
    key = 'pt={}'.format(p_t)
    word_features.append(key)
    key = 'ppt={}^{}'.format(pp_t, p_t)
    word_features.append(key)
    # Generate next words and words that appeared before in the sentence.
    if word_ind > 0:
        key = 'pw={}'.format(words[word_ind - 1])
        word_features.append(key)
    if word_ind > 1:
        key = 'ppw={}'.format(words[word_ind - 2])
        word_features.append(key)
    if word_ind < len(words) - 1:
        key = 'nw={}'.format(words[word_ind + 1])
        word_features.append(key)
    if word_ind < len(words) - 2:
        key = 'nnw={}'.format(words[word_ind + 2])
        word_features.append(key)
    return word_features


def create_all_features(train_raw_data, word_counters):
    """
    Create all features from the corpus.
    :param train_raw_data: Train raw data extracted before.
    :param word_counters: Word counters.
    :return: List of all possible features of each word.
    """
    all_corpus_features = []
    for words, tags in train_raw_data:
        for ind, word in enumerate(words):
            # Assign for each word previous and previous of previous tags.
            if ind > 1:
                pp_t = tags[ind - 2]
                p_t = tags[ind - 1]
            elif ind > 0:
                pp_t = START_SYMBOL
                p_t = tags[ind - 1]
            else:
                pp_t = START_SYMBOL
                p_t = START_SYMBOL
            is_rare = False
            # Generate word features as the word in not rare.
            features_list = generate_word_features(is_rare, p_t, pp_t, word, ind, words)
            all_corpus_features.append('{} {}'.format(tags[ind], ' '.join(features_list)))
            # Check rarity of the word. And generate respectively features as rare word.
            if word_counters[word] == RARE_WORD_MAX_FREQ:
                is_rare = True
                features_list = generate_word_features(is_rare, p_t, pp_t, word, ind, words)
                all_corpus_features.append('{} {}'.format(tags[ind], ' '.join(features_list)))
    return all_corpus_features


def extract(train_fname, features_fname):
    """
    Start extraction all raw data from training corpus and create all possible features in the corpus.
    :param train_fname: Train file name.
    :param features_fname: Feature files name.
    :return:
    """
    start = time.time()

    print 'starting to extract raw data...'
    train_raw_data, word_counters = extract_raw_data(train_fname)
    print 'creating features...'
    all_corpus_features = create_all_features(train_raw_data, word_counters)
    print 'writing the features to the file...'
    output(features_fname, all_corpus_features)

    end = time.time()
    print('time took to train: ' + str(end - start) + ' sec')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'need at least 2 arguments.'
        exit(1)

    train_file_name = sys.argv[1]
    features_file_name = sys.argv[2]

    extract(train_file_name, features_file_name)
