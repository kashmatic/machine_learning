from math import sqrt
import numpy as np
from collections import Counter
import random
import pandas as pd

## Eucliedean can define a radius to avoid going through all datapoints

def k_nearest_neighbors(data, predict, k=3):
    distances = []
    for group in data:
        for feature in data[group]:
            euclidean_distance = np.linalg.norm(np.array(feature)-np.array(predict))
            distances.append([euclidean_distance, group])

    ## Get the top k values
    votes = [i[1] for i in sorted(distances)[:k]]

    ## Get the most_common key
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

df = pd.read_csv('breast-cancer-wisconsin.data')
## Replace the missing date (?) with rediculously low number
df.replace('?', -99999, inplace=True)
## We do not need the id column, because it doesnt contibute anything
df.drop(['id'], 1, inplace=True)

full_data = df.astype(float).values.tolist()

random.shuffle(full_data)

## Split the data to train and test
test_size = 0.2

train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}

train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]

for i in train_data:
    ## use the last column 2 or 4 to assign to the group
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    ## use the last column 2 or 4 to assign to the group
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=551)
        if group == vote:
            correct += 1
        total += 1

print('Accuracy: ', correct/total)
