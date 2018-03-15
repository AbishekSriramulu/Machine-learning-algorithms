import numpy as np
from sklearn import neighbors, model_selection, svm
import pandas as pd

df = pd.read_csv('9TL0B17S77.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

x = np.array(df.drop(['class'],1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.2)

clf = svm.SVC()
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print(accuracy)

example_measure = np.array([[4,2,1,1,1,2,3,2,1], [4,2,2,2,2,2,3,2,1]])
example_measure = example_measure.reshape(len(example_measure),-1)

pred = clf.predict(example_measure)
print(pred)