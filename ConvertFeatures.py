import sys
import time
from collections import defaultdict


STUDENT = {'name': 'Alex Kartun_Ofir Sharon',
           'ID': '324429216_204717664'}


def features_to_vec(features, tags_map, features_map, word_to_tags_map):
    """
    Update map of tags to indexes and map of features to indexes and word to tags map if needed.
    Convert features to vector of indexes by assigning string feature to his index in the features map.
    :param features: List of word features.
    :param tags_map: Map of tags to indexes.
    :param features_map: Map of features to indexes.
    :param word_to_tags_map: Word to his tags map.
    :return: Vector of features.
    """
    feature_vec = []
    tag = features[0]

    for feature in features[1:]:
        if feature.startswith('form='):
            word_to_tags_map[feature.split('=')[1]].add(tag)
        if feature not in features_map:
            features_map[feature] = len(features_map) + 1
        feature_vec.append(features_map[feature])

    feature_vec = sorted(feature_vec)
    if tag not in tags_map:
        tags_map[tag] = len(tags_map)

    feature_vec.insert(0, tags_map[tag])

    return feature_vec


def features_to_vecs(features_data):
    """
    Convert all corpus features to vectors.
    :param features_data: List of all features.
    :return:
    """
    tags_map = defaultdict(int)
    word_to_tags_map = defaultdict(set)
    features_map = defaultdict(int)
    feature_vecs = []
    for features_line in features_data:
        features = features_line.strip().split()
        feature_vec = features_to_vec(features, tags_map, features_map, word_to_tags_map)
        feature_vecs.append(feature_vec)

    return feature_vecs, features_map, tags_map, word_to_tags_map


def output(feature_vecs_fname, feature_map_fname, feature_vecs, features_map, tags_map, word_to_tags_map):
    """
    Output the feature vecs and feature map/tags/map/word to tags map to the files.
    :param feature_vecs_fname: Feature vector file name.
    :param feature_map_fname: Feature map file name.
    :param feature_vecs: List of all feature vectors.
    :param features_map: Features to indexes map.
    :param tags_map: Tags to indexes map.
    :param word_to_tags_map: Word to tags map.
    :return:
    """
    output_vecs = ['{} {}'.format(feature_vec[0], ' '.join('{}:1'.format(feature)for feature in feature_vec[1:]))
                   for feature_vec in feature_vecs]
    with open(feature_vecs_fname, 'w') as f:
        f.write('\n'.join(output_vecs))

    # Write all maps to feature map file separated by '^' between tags to word to tags map.
    # And by '^^' between word to tags map to features map. The format of each entry is 'key:value'.
    output_tags_map = ['{} {}\n'.format(k, v) for k, v in tags_map.items()]
    output_word_to_tags_map = ['{}^{}\n'.format(k, ' '.join(t for t in v)) for k, v in word_to_tags_map.items()]
    output_features_map = ['{} {}'.format(k, v) for k, v in features_map.items()]
    with open(feature_map_fname, 'w') as f:
        f.write(''.join(output_tags_map))
        f.write('^\n')
        f.write(''.join(output_word_to_tags_map))
        f.write('^^\n')
        f.write('\n'.join(output_features_map))


def convert(features_fname, feature_vecs_fname, feature_map_fname):
    """
    Converting all string features to indexes vectors of features.
    :param features_fname: Features file name.
    :param feature_vecs_fname: Feature vector file name.
    :param feature_map_fname: Feature map file name.
    :return:
    """
    start = time.time()

    print 'starting to extract raw data...'
    with open(features_fname, 'r') as f:
        features_data = f.readlines()
    print 'creating feature vecs and maps...'
    feature_vecs, features_map, tags_map, word_to_tags_map = features_to_vecs(features_data)
    print 'writing the feature vecs and feature/tag maps to the file...'
    output(feature_vecs_fname, feature_map_fname, feature_vecs, features_map, tags_map, word_to_tags_map)

    end = time.time()
    print('time took to train: ' + str(end - start) + ' sec')


def load_model(feature_map_fname):
    """
    Load the model from features map file.
    :param feature_map_fname: Features map file name.
    :return: Tags set, index to tag map, word to tags map and features map.
    """
    features_map = defaultdict(int)
    ind_to_tags_map = defaultdict(str)
    tags_set = set()
    word_to_tags_map = defaultdict(set)

    with open(feature_map_fname, 'r') as f:
        all_data = f.readlines()
    # Booleans to separate the reading process from the file.
    is_features_section = False
    is_words_to_tags_section = False
    is_tags_section = True
    for ind, line in enumerate(all_data):
        line = line.strip()
        if line == '^':
            is_words_to_tags_section = True
            is_tags_section = False
            continue
        if line == '^^':
            is_features_section = True
            is_words_to_tags_section = False
            continue
        if is_tags_section:
            key, value = line.split()
            tags_set.add(key)
            ind_to_tags_map[int(value)] = key
        if is_words_to_tags_section:
            key, value = line.split('^')
            word_to_tags_map[key] = value.split()
        if is_features_section:
            key, value = line.split()
            features_map[key] = int(value)
    return features_map, tags_set, ind_to_tags_map, word_to_tags_map


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print 'need at least 3 arguments.'
        exit(1)

    features_file_name = sys.argv[1]
    feature_vecs_file_name = sys.argv[2]
    feature_map_file_name = sys.argv[3]

    convert(features_file_name, feature_vecs_file_name, feature_map_file_name)
