# -*-coding:utf-8 -*-
__author__ = 'Administrator'

import numpy as np

class WeakClassifier:

    def __init__(self):
        self.dim = 0
        self.val = 0.0
        self.symb = 'lt'
        self.features_num = 0

    def train(self, x_train, y_train, W, steps=100):

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        W = np.array(W)

        min_error = np.inf
        sample_num = x_train.shape[0]
        features_num = x_train.shape[1]  # 对每个属性都进行测试，最后得到误差最小的属性，也即是维度
        self.features_num = features_num

        for ii in range(features_num):

            # 获取ii维度上的最小最大值，根据stepsize进行遍历判断最佳分割值
            dim_min = np.min(x_train[:, ii])
            dim_max = np.max(x_train[:, ii])
            stepsize = (dim_max - dim_min) / steps

            for value in np.arange(dim_min, dim_max, stepsize):

                for symb in ['lt', 'gt']:
                    tp_est = np.ones(y_train.shape)
                    tp_x = x_train[:, ii]
                    if symb == 'lt':
                        tp_est[tp_x < value] = -1
                    else:
                        tp_est[tp_x > value] = -1
                    error = np.sum((tp_est != y_train) * W)
                    if error < min_error:
                        min_error = error
                        self.dim = ii
                        self.val = value
                        self.symb = symb
        return min_error

    def pred(self, x_test):
        x_test = np.array(x_test).reshape(-1, self.features_num)
        tp_est = np.ones((x_test.shape[0], 1)).flatten(1)
        if self.symb == 'lt':
            tp_est[x_test[:, self.dim] < self.val] = -1
        else:
            tp_est[x_test[:, self.dim] > self.val] = -1
        return tp_est


class AdaBoostClassifier:

    def __init__(self, n_estimator=4, weakClf=WeakClassifier):
        self.n_estimator = n_estimator
        self.real_n_estimator = 0  # 记录真正用到的弱分类器，因为有可能在n_estimator之内就找到了零误差的分类器
        self.weakClf = WeakClassifier
        self.X = []
        self.Y = []
        self.W = []  # 存放每个数据点的权重
        self.G = []
        self.alpha = []
        self.features_num = 0

    def train(self, x_train, y_train):
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        self.X = x_train
        self.features_num = x_train.shape[1]
        self.Y = y_train
        self.W = np.ones(y_train.shape) / len(y_train)
        total_est = np.zeros(y_train.shape)  # 记录以用于判断当前所有分类器分类误差是否为0
        for ii in range(self.n_estimator):
            tp_G = WeakClassifier()
            err = tp_G.train(x_train, y_train, self.W)
            tp_alpha = 0.5 * np.log((1 - err) / max(err, 1e-16))  # 用1e-16以防除以0
            self.G.append(tp_G)
            self.alpha.append(tp_alpha)
            tp_est = tp_G.pred(x_train)
            Z = self.W * np.exp(-self.alpha[ii] * y_train * tp_est)
            self.W = (Z / Z.sum()).reshape(len(y_train), -1).flatten(1)
            self.real_n_estimator += 1
            total_est += total_est + tp_est * tp_alpha
            pred_y = np.sign(total_est)
            if (pred_y != y_train).sum() == 0:
                print self.real_n_estimator, ' weak estimators is enough !'
                break

    def pred(self, x_test):

        x_test = np.array(x_test).reshape(-1, self.features_num)
        total_est = np.zeros((x_test.shape[0], 1))

        for ii in range(self.real_n_estimator):
            total_est += total_est + self.G[ii].pred(x_test) * self.alpha[ii]

        pred_y = np.sign(total_est)
        return pred_y


dataMat = np.matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

ada = AdaBoostClassifier()
ada.train(dataMat, classLabels)
print ada.pred([1, 2.1])



