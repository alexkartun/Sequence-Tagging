import sys
STUDENT = {'name': 'Alex Kartun_Ofir Sharon',
           'ID': '324429216_204717664'}


def calc_accuracy(output_file, test_file):
    """
    Calculate accuracy of right tags divided by total.
    :param output_file: Output file name
    :param test_file: Test file name.
    :return: Accuracy.
    """
    total_tags = right_tags = 0.0

    output_tags = extract_tags(output_file)
    test_tags = extract_tags(test_file)

    for o_tags, t_tags in zip(output_tags, test_tags):
        for o_tag, t_tag in zip(o_tags, t_tags):
            if o_tag == t_tag:
                right_tags += 1
            total_tags += 1
    return right_tags / total_tags


def extract_tags(file_name):
    """
    Extraction all the tags from the file.
    :param file_name: File name.
    :return: ist of all tags.
    """
    tags = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            tags.append([part.split('/')[-1] for part in parts])
    return tags


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print

    dev_output_file_name = sys.argv[1]
    dev_test_file_name = sys.argv[2]

    acc = calc_accuracy(dev_output_file_name, dev_test_file_name)
    print "acc: {}".format(acc)
