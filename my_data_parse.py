import pandas as pd
import numpy as np
import random


def parse_mydata():
    features = []
    with open('data_15000.csv') as f:
        lines = f.readlines()
        for line in lines:
            splitted = line.split(';')
            #featureset = list(map(int,splitted[:9414]))
            featureset = [int(float(x)) for x in splitted[:len(splitted)-1]]
            label = splitted[-1]
            hotlable = []
            if label == '1':
                hotlable =[1,0]
            else:
                hotlable = [0,1]
            features.append([featureset, hotlable])
    return features
    #     for line in lines:
    #         splitted = list(map(int,line.split(';')))
    #         featureset = splitted[:len(splitted)-1]
    #         label=splitted[-1]
    #         hotlable = []
    #         if label==1:
    #             hotlable=[1,0]
    #         else:
    #             hotlable=[0,1]
    #         features.append([featureset, hotlable])
    # return features


def bring_processed():
    features = parse_mydata()
    random.shuffle(features)
    features = np.array(features)

    testing_size = int(0.2 *len(features))

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x, train_y, test_x, test_y
if __name__=='__main__':
    print(bring_processed())
    #parse_mydata()
