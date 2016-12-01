# -*-coding:utf-8-*-
_author_ = "shying"
from ML_Util.distance_metric import get_distance
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import operator
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4)


def classify(x, test_data, labels, k):
    dis = get_distance(x, test_data)
    dis_indices = dis.argsort()  # sort the distances and get the indices

    # use a dictionary to record the first k points' label
    labelRecord = {}

    # get the label of x according to the first k points
    for i in range(k):
        test_label = labels[dis_indices[i]]
        labelRecord[test_label] = labelRecord.get(test_label, 0) + 1

    # sort the dict according to the value
    sortedRecord = sorted(labelRecord.iteritems(), key=operator.itemgetter(1), reverse=True)

    return sortedRecord[0][0]

# compare to the KNN in sklearn
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(x_test, y_test)
difNum = 0  # record the number which our KNN's prediction is different to the sklearn's
for sample in x_train:
    a = classify(sample, x_test, y_test, 5)
    b = clf.predict(sample.reshape(1, -1))
    if a != b:
        difNum += 1

print difNum
