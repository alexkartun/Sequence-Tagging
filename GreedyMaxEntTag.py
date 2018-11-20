import sys
import time
import ExtractFeatures as extract
import ConvertFeatures as convert
import numpy as np
from liblin import LiblinearLogregPredictor


STUDENT = {'name': 'Alex Kartun_Ofir Sharon',
           'ID': '324429216_204717664'}


def features_to_vec(word_features, f_map):
    """
    Convert word features to vector of features indexes whose liblinear model can understand.
    :param word_features: Word features to convert.
    :param f_map: Features map.
    :return: Vector of features.
    """
    features_vec = []
    for feature in word_features:
        if feature in features_map:
            features_vec.append(f_map[feature])

    features_vec = sorted(features_vec)
    return features_vec


def tagging(inp_dev_data, f_map, i_t_map, word_t_map, lib_model):
    """
    Tagger algorithm.
    :param inp_dev_data: List of all dev words to be tagged.
    :param f_map: Features map.
    :param i_t_map: Index to tags map.
    :param word_t_map: Word to tags map.
    :param lib_model: Liblinear model util.
    :return: List of List of all vectors x as words and y as tags respectively.
    """
    out_dev_data = []

    for line in inp_dev_data:
        x = [word for word in line.strip().split(" ")]
        y = predict_greedy(x, f_map, i_t_map, word_t_map, lib_model)
        out_dev_data.append(' '.join(['{}/{}'.format(word, tag) for word, tag in zip(x, y)]))
    return out_dev_data


def predict_greedy(x, f_map, i_t_map, word_t_map, lib_model):
    """
    Greedy algorithm of decoding that done prediction's of liblinear mode.
    :param x: X vector of all words to be tagged.
    :param f_map: Features map.
    :param i_t_map: Index to tags map.
    :param word_t_map: Word to tags map.
    :param lib_model: Liblinear model.
    :return: Return best predicted list of tags for each word in x respectively.
    """
    y = []
    pp_t = extract.START_SYMBOL
    p_t = extract.START_SYMBOL
    for ind, curr_word in enumerate(x):
        # Check if current word is seen in training corpus.
        if curr_word not in word_t_map:
            is_rare = True
        else:
            is_rare = False
        # Generate word features depends of his rarity status.
        word_features = extract.generate_word_features(is_rare, p_t, pp_t, curr_word, ind, x)
        features_vec = features_to_vec(word_features, f_map)
        tag = get_greedy_tag(features_vec, i_t_map, lib_model)
        y.append(tag)
        pp_t = p_t
        p_t = tag
    return y


def get_greedy_tag(features_vec, i_t_map, lib_model):
    """
    Get greedy best tag from prediction of the model on features vector.
    :param features_vec: Features vector.
    :param i_t_map: Index to tags map.
    :param lib_model: Liblinear model for prediction.
    :return: A predicted tag of the model.
    """
    scores = lib_model.predict(features_vec)
    index = np.argmax(scores)
    max_tag = i_t_map[int(index)]
    return max_tag


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print 'need at least 4 arguments.'
        exit(1)

    input_dev_file_name = sys.argv[1]
    model_file_name = sys.argv[2]
    feature_map_file_name = sys.argv[3]
    output_file_name = sys.argv[4]

    start = time.time()

    print 'loading a model...'
    lib = LiblinearLogregPredictor(model_file_name)
    features_map, tags_set, ind_to_tags_map, word_to_tags_map = convert.load_model(feature_map_file_name)

    print 'extracting dev data...'
    with open(input_dev_file_name, 'r') as input_file:
        input_dev_data = input_file.readlines()
    print 'starting to predict and greedy decoding...'
    output_dev_data = tagging(input_dev_data, features_map, ind_to_tags_map, word_to_tags_map, lib)
    print 'writing the tagged output to the file...'
    with open(output_file_name, 'w') as output_file:
        output_file.write('\n'.join(output_dev_data))

    end = time.time()
    print('time took to predict and greedy decoding: ' + str(end - start) + ' sec')
