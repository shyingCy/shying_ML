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
    label.append(field[0])
f.close()


def lr_fun(X):
    return 1.0 / (1+np.exp(-X))

num = data.shape[0]
features_num = data.shape[1]
w = np.zeros(3)

# process the label data, change the -1 label to 0
for i in range(num):
    if label[i] == -1.0:
        label[i] = 0.0

# first iteration
z = np.dot(data, w)
h = lr_fun(z)
cost_function_pre = 1
cost_function_pre = sum(label * np.log(h) + (np.ones(num)-label) * np.log(np.ones(num)-h))

G = np.zeros(features_num)  # first derivation
H = np.zeros((features_num, features_num))  # second derivation
for i in range(features_num):
    dif = label - h
    G[i] = sum(dif * data[:, i])/num

    const_sum = h * (np.ones(num) - h)
    for j in range(features_num):
        H[i][j] = -sum(const_sum * data[:, i] * data[:, j]) / num

# update w
w = w - np.dot(np.linalg.inv(H), G)

# second iteration
z = np.dot(data, w)
h = lr_fun(z)
cost_function = sum(label * np.log(h) + (np.ones(num)-label) * np.log(np.ones(num)-h))

thred = 1e-5
max_iter = 1500
iters = 2
while abs(cost_function - cost_function_pre) >= thred and iters < max_iter:
    iters += 1
    cost_function_pre = cost_function
    G = np.zeros(features_num)  # first derivation
    H = np.zeros((features_num, features_num))  # second derivation
    for i in range(features_num):
        dif = label - h
        G[i] = sum(dif * data[:, i])/num

        const_sum = h * (np.ones(num) - h)
        for j in range(features_num):
            H[i][j] = -sum(const_sum * data[:, i] * data[:, j]) / num
    # update w
    w = w - np.dot(np.linalg.inv(H), G)
    z = np.dot(data, w)
    h = lr_fun(z)
    cost_function = sum(label * np.log(h) + (np.ones(num)-label) * np.log(np.ones(num)-h))
    print 'iteration ', iters, ':', abs(cost_function - cost_function_pre)

print 'w: ', w
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

# calculate the accuracy
acc = 0.0
for i in range(num):
    pred = np.dot(data[i, :], w)
    if pred < 0.5:
        pred = 0.0
    else:
        pred = 1.0
    if pred == label[i]:
        acc += 1.0
acc /= num
print 'data accuracy is ', acc

