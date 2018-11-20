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


def tagging(inp_dev_data, f_map, tags_s, word_t_map, lib_model):
    """
    Tagger algorithm.
    :param inp_dev_data: List of all dev words to be tagged.
    :param f_map: Features map.
    :param tags_s: Tags set.
    :param word_t_map: Word to tags map.
    :param lib_model: Liblinear model util.
    :return: List of List of all vectors x as words and y as tags respectively.
    """
    out_dev_data = []

    for line in inp_dev_data:
        x = [word for word in line.strip().split()]
        y = predict_viterbi(x, f_map, tags_s, word_t_map, lib_model)
        out_dev_data.append(' '.join(['{}/{}'.format(word, tag) for word, tag in zip(x, y)]))
    return out_dev_data


def predict_viterbi(x, f_map, tags_s, word_t_map, lib_model):
    """
    For each word in vector x predict his tag. Prediction is done by viterbi algorithm. Check all tags options/globally.
    :param x: X vector of all words to be tagged.
    :param f_map: Features map.
    :param tags_s: Tags set.
    :param word_t_map: Word to tags map.
    :param lib_model: Liblinear model.
    :return: Return best predicted list of tags for each word in x respectively.
    """
    y = []
    v = [{(extract.START_SYMBOL, extract.START_SYMBOL): 0.0}]
    bp = []
    for ind, word in enumerate(x):
        # Check if word was seen in the corpus.
        if word not in word_t_map:
            is_rare = True
            available_tags = tags_s
        else:
            is_rare = False
            # Pruning of tags to lower amount of possible tags for this word.
            available_tags = word_t_map[word]

        max_score = {}
        max_tags = {}
        # Calculate for each word best scores/probabilities and best tags for each word.
        for pp_t, p_t in v[ind]:
            for curr_tag in available_tags:
                word_features = extract.generate_word_features(is_rare, p_t, pp_t, word, ind, x)
                features_vec = features_to_vec(word_features, f_map)
                scores = lib_model.predict(features_vec)
                score = np.amax(scores)
                if (p_t, curr_tag) not in max_score or score > max_score[(p_t, curr_tag)]:
                    max_score[(p_t, curr_tag)] = score
                    max_tags[(p_t, curr_tag)] = pp_t

        v.append(max_score)
        bp.append(max_tags)
    # Calculate last 2 best tags.
    max_score = float("-inf")
    prev_last_tag, last_tag = None, None
    for prev_t, curr_t in v[len(x)]:
        score = v[len(x)][(prev_t, curr_t)]
        if score > max_score:
            max_score = score
            last_tag = curr_t
            prev_last_tag = prev_t

    y.append(last_tag)
    if len(x) > 1:
        y.append(prev_last_tag)

    prev_t = last_tag
    prev_prev_t = prev_last_tag
    # By backtracking extract all the path of best tags for each word starting by last 2 tags we calculated above.
    for i in range(len(v) - 2, 1, -1):
        curr_t = bp[i][(prev_prev_t, prev_t)]
        y.append(curr_t)
        prev_t = prev_prev_t
        prev_prev_t = curr_t
    y = reversed(y)
    return y


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
    print 'starting to predict and viterbi decoding...'
    output_dev_data = tagging(input_dev_data, features_map, tags_set, word_to_tags_map, lib)
    print 'writing the tagged output to the file...'
    with open(output_file_name, 'w') as output_file:
        output_file.write('\n'.join(output_dev_data))

    end = time.time()
    print('time took to predict and viterbi decoding: ' + str(end - start) + ' sec')
