import numpy as np
from math import  sqrt
import  matplotlib.pyplot as plt
from matplotlib import  style
import warnings
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

dataset = {'k': [[1,2], [2,3], [3,1]], 'r' : [[6,5], [7,7], [8,6]]}
new_features = [5,7]



def k_nearest_neighbours (data,predict, k = 3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distance = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distance.append([euclidean_distance,group])

    votes = [i[1]  for i in sorted(distance) [:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

result = k_nearest_neighbours(dataset,new_features, k=3)
print(result)

[[ plt.scatter(ii[0], ii[1], s = 100, color= i) for ii in dataset[i]] for i in dataset]
plt.show()