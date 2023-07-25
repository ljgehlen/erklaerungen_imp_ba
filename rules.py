import pandas as pd
import numpy as np


def get_grad(model, te_data: tuple, optimizer, loss_fn):
    print(' ')


def encode_test(vocab_size, test_data):y
    frame = np.zeros(((test_data.shape)[0], vocab_size))
    test = 0
    for instance in test_data:
        for value in instance:
            #frame[test][value.item()] += 1
            frame[test][value.item()] = 1
        test += 1
    print(frame)


def save_rules(vocab, test_data):

    features = []
    for i in range(len(vocab)):
        features.append(str(i))

    #dataframe = pd.DataFrame(test_data, columns=features)
    print(test_data.shape)

    for i in test_data:
        print(i.shape)
        break