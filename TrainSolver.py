import sys

STUDENT = {'name': 'Alex Kartun_Ofir Sharon',
           'ID': '324429216_204717664'}


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'need at least 2 arguments.'
        exit(1)

    feature_vecs_file_name = sys.argv[1]
    model_file_name = sys.argv[2]

    from subprocess import call
    # If there is only 3 arguments run the linear model with our regulization.
    if len(sys.argv) == 3:
        send = 'java -cp liblinear.jar de.bwaldvogel.liblinear.Train -s 0 {} {}'.format(feature_vecs_file_name,
                                                                                        model_file_name)
    # If there is only 5 arguments run the linear model with regulization as passed as argument.
    elif len(sys.argv) == 5:
        regulization_arguments = sys.argv[3]
        regulization_value = sys.argv[4]
        send = 'java -cp liblinear.jar de.bwaldvogel.liblinear.Train -s 0 {} {} {} {}'.format(regulization_arguments,
                                                                                              regulization_value,
                                                                                              feature_vecs_file_name,
                                                                                              model_file_name)
    formatted = send.split()
    call(formatted)
