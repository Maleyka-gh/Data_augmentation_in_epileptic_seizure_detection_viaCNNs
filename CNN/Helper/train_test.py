import random
import pandas as pd

def randomize_train_test(dictionary):
    condition = True
    train = test = explanation = None
    while condition:
        train = list()
        test = list()
        explanation = {'train': list(), 'test': list()}
        for key, val in dictionary.items():
            if bool(random.getrandbits(1)):
                train.append(val)
                explanation['train'].append(key)
            else:
                test.append(val)
                explanation['test'].append(key)
        condition = len(train) == 0 or len(test) == 0
    return train, test, explanation


def split_x_y(dictionary):
    df = pd.concat(dictionary)
    y_df = df['class']
    # temp encoding:
    #from sklearn import preprocessing
    #le = preprocessing.LabelEncoder()
    #le.fit(y_df)
    #print('PAUSE')


    X_df = df.drop(columns='class')
    return X_df, y_df