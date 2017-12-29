#import pandas as pd
import numpy as np
import random
import time

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

def make_feature_row(line):
    splitted = line.split(';')
    #featureset = list(map(int,splitted[:9414]))
    featureset = [int(float(x)) for x in splitted[:len(splitted)-1]]
    label = int(float(splitted[-1].rstrip()))
    #print(label)
    hotlable = []
    if label == 1:
        #print('o', end=',')
        hotlable =[1,0]
    else:
        #print('t', end=',')
        hotlable = [0,1]
    #print(label, hotlable)
    return [featureset, hotlable]

def get_test_data():
    features = []
    with open('test_data.csv') as f:
        features = [make_feature_row(l) for l in f]
    features = np.array(features)
    test_x = list(features[:,0])
    test_y = list(features[:,1])
    print(len(test_x[0]))
    return test_x, test_y

def get_train_data_batch(n):
    features = []
    with open('train_data.csv') as f:
        ct=0
        for l in f:
            features.append(make_feature_row(l))
            ct+=1
            if ct%n==0:
                features = np.array(features)
                train_x = list(features[:,0])
                train_y = list(features[:,1])
                if ct%1000==0:
                    print("giving data", ct)
                yield train_x, train_y
                features = []

if __name__=='__main__':
    # batch_size = 100
    # for j in range(2):
    #     train_data_batch_gen = get_train_data_batch(batch_size)
    #     i=0
    #     while i < 24000:
    #         start = i
    #         end = i+batch_size
    #         train_x, train_y = train_data_batch_gen.__next__()
    #         i+=batch_size
    #print(bring_processed())
    #parse_mydata()
    #time1 = time.time()
    #test_x, test_y = get_test_data()
    # train_batch_gen=get_train_data_batch(10)
    # for i in range(100):
    #     nnn = train_batch_gen.__next__()
    td = get_test_data()
    #pass
