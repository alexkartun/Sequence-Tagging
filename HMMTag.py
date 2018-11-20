import sys
import time
import math
import MLETrain as mle_train


STUDENT = {'name': 'Alex Kartun_Ofir Sharon',
           'ID': '324429216_204717664'}


def tagging(inp_dev_data, trans_c, emiss_c, word_tags_map, interp_weights):
    """
    Tagger.
    :param inp_dev_data: Dev data to predict on.
    :param trans_c: Transaction counts.
    :param emiss_c: Emission counts.
    :param word_tags_map: Word to tags map.
    :param interp_weights: Interpolation weights.
    :return: List of all vectors x as words and y as tags respectively.
    """
    out_dev_data = []

    for line in inp_dev_data:
        x = [word for word in line.strip().split(" ")]
        y = predict_viterbi(x, trans_c, emiss_c, word_tags_map, interp_weights)
        out_dev_data.append(' '.join(['{}/{}'.format(word, tag) for word, tag in zip(x, y)]))
    return out_dev_data


def predict_viterbi(x, trans_c, emiss_c, word_tags_map, interp_weights):
    """
    For each word in vector x predict his tag. Prediction is done by viterbi algorithm. Check all tags options/globally.
    :param x: X vector of words.
    :param trans_c: Transaction counts.
    :param emiss_c: Emission counts.
    :param word_tags_map: Word to tags map.
    :param interp_weights: Interpolation weights.
    :return: Vector of all tags respectively to words in vector x.
    """
    y = []
    v = [{(mle_train.START_SYMBOL, mle_train.START_SYMBOL): 0.0}]
    bp = []
    for ind, word in enumerate(x):
        # Convert word if it was'nt seen in the corpus, to signature word.
        if word not in word_tags_map:
            word = mle_train.subcategorize(word)
        # Pruning of tags to lower amount of possible tags for this word.
        available_tags = word_tags_map[word]

        max_score = {}
        max_tags = {}
        # Calculate for each word best scores/probabilities and best tags for each word.
        for pp_t, p_t in v[ind]:
            for curr_tag in available_tags:
                score = get_score(word, curr_tag, p_t, pp_t, trans_c, emiss_c, interp_weights)
                score += v[ind][(pp_t, p_t)]

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
    # Reverse the path.
    y = reversed(y)
    return y


def get_score(word, curr_tag, p_t, pp_t, trans_c, emiss_c, interp_weights):
    """
    Calculate probability. Prob = e_prob + q_prob.
    :param word: Curr word.
    :param curr_tag: Curr tag.
    :param p_t: Previous tag.
    :param pp_t: Previous of previous tag.
    :param trans_c: Transaction counts.
    :param emiss_c: Emission counts.
    :param interp_weights: Interpolation weights.
    :return:
    """
    e = mle_train.get_e(word, curr_tag, emiss_c, trans_c)
    q = mle_train.get_q(pp_t, p_t, curr_tag, trans_c, interp_weights)
    if e != 0 and q != 0:
        score = math.log(e) + math.log(q)
    else:
        score = float('-inf')
    return score


if __name__ == '__main__':
    if len(sys.argv) < 6:
        print 'need at least 5 arguments.'
        exit(1)

    input_dev_file_name = sys.argv[1]
    q_mle_file_name = sys.argv[2]
    e_mle_file_name = sys.argv[3]
    output_file_name = sys.argv[4]
    extra_file_name = sys.argv[5]
    start = time.time()

    print 'loading a model...'
    transaction_c, emission_c, tag_set, word_to_tags_map, interpolation_weights = mle_train.load_model(q_mle_file_name,
                                                                                                       e_mle_file_name,
                                                                                                       extra_file_name)
    print 'extracting dev data...'
    with open(input_dev_file_name, 'r') as input_file:
        input_dev_data = input_file.readlines()
    print 'starting to predict and viterbi decoding...'
    output_dev_data = tagging(input_dev_data, transaction_c, emission_c, word_to_tags_map, interpolation_weights)
    print 'writing the tagged output to the file...'
    with open(output_file_name, 'w') as output_file:
        output_file.write('\n'.join(output_dev_data))

    end = time.time()
    print('time took to predict and viterbi decoding: ' + str(end - start) + ' sec')
