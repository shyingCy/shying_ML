#-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
data = []
f = open("logistic_x.txt", "r")
for line in f:
    field = line.split()
    field = [float(x) for x in field]
    data.append(field)
f.close()

data = np.array(data)
data = np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)

label = []
f = open("logistic_y.txt", "r")
for line in f:
    field = line.split()
    field = [float(x) for x in field]
    label.append(field)
f.close()
"""
# 这一段注释用sklearn里的波斯顿房价数据集，用了所有维度，也可以试试两个维度的
from sklearn import datasets
boston = datasets.load_boston()
data = boston.data
data = np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)

label = boston.target

def preprocess(data):
    for ii in range(data.shape[1]):
        divD = np.max(data[:, ii] - np.min(data[:, ii]))
        if divD != 0:
            data[:, ii] = data[:, ii] / divD
    return data

data = preprocess(data)
label = preprocess(label.reshape(1, len(label))).transpose()
"""
features_num = data.shape[1]
w = np.array([1.0 / (features_num)] * (features_num))

alpha = 0.05
J = 1
for i in range(1500):
    w_old = w
    J = 0
    for w_num in range(len(w)):
        loss = 0
        for ii in range(data.shape[0]):
            loss += (np.dot(data[ii], w_old) - label[ii]) * data[ii][w_num]
        w[w_num] -= alpha * (loss / data.shape[0])

    for ii in range(data.shape[0]):
        J += (abs(np.dot(data[ii], w) - label[ii]) / (2.0 * data.shape[0]))
    print J
    # J /= (2.0 * data.shape[0])

print 'w:', w
x = np.arange(np.min(data[:, 1]), np.max(data[:, 1]), 0.01)
y = -(w[0]/w[2]) - (w[1]/w[2]) * x
plt.figure()
label_type = np.unique(label)
for ii in range(data.shape[0]):
    if label[ii] == label_type[0]:
        plt.scatter(data[ii, 1], data[ii, 2], c='g')
    else:
        plt.scatter(data[ii, 1], data[ii, 2], c='r')
plt.plot(x, y, label='train line')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

print np.dot(data[0, :], w)
print np.dot(data[1, :], w)
print np.dot(data[2, :], w)
