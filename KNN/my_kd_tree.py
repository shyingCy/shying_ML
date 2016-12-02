# -*-coding:utf-8-*-
_author_ = "shying"

import numpy as np
from ML_Util.distance_metric import get_distance

class KD_node:
    def __init__(self, point=None, split=None, LL=None, RR=None):
        '''

        :param point:数据点
        :param split:分割维度
        :param LL:结点的左儿子
        :param RR:结点的右儿子
        :return:
        '''
        self.point = point
        self.split = split
        self.left = LL
        self.right = RR


class KD_Tree:
    root = []

    def __init__(self, data=None):
        self.data = data

    def build(self, tree, dataList):

        LEN = len(dataList)
        if LEN == 0:
            return

        features_num = len(dataList[0])
        dataList = np.array(dataList) # 转化为numpy类型便于处理
        var = 0
        split = 0 # 分割的维度
        # 根据方差最大的维度选择split域
        for i in range(features_num):
            tmp_var = np.var(dataList[:, i])
            if tmp_var > var:
                var = tmp_var
                split = i

        # 根据分割的维度对数据进行排序
        dataList = dataList[np.argsort(dataList[:, split])]
        # 选择LEN/2 的点为分割点
        point = dataList[LEN/2]
        tree = KD_node(point, split)
        tree.left = self.build(tree.left, dataList[0: LEN/2])
        tree.right = self.build(tree.right, dataList[LEN/2 + 1: LEN])
        return tree

    def buildTree(self):
        self.root = self.build(self.root, self.data)

    def tree_preorder(self, tree):

        print tree.point

        if tree.left:
            self.tree_preorder(tree.left)

        if tree.right:
            self.tree_preorder(tree.right)

    def preorder(self):
        self.tree_preorder(self.root)

    # 返回距离findPoint最近的点以及距离
    def findNearest(self, findPoint):

        # NP存储离findPoint最近的点，min_dist存储最近距离
        NP = self.root.point
        min_dist = get_distance(NP, findPoint)[0]

        temp_root = self.root
        nodeList = []  # 存储经过的结点
        while temp_root:

            nodeList.append(temp_root)
            dd = get_distance(temp_root.point, findPoint)[0]
            if dd < min_dist:
                NP = temp_root.point
                min_dist = dd
            # 获取当前结点分割域
            split = temp_root.split
            if findPoint[split] <= temp_root.point[split]:
                temp_root = temp_root.left
            else:
                temp_root = temp_root.right

        # 对经历过的结点进行回溯检查
        while nodeList:
            back_root = nodeList.pop()
            split = back_root.split
            if abs(findPoint[split] - back_root.point[split]) < min_dist:
                if findPoint[split] < back_root.point[split]:
                    temp_root = back_root.right
                else:
                    temp_root = back_root.left

            if temp_root:
                nodeList.append(temp_root)
                dd = get_distance(findPoint, temp_root.point)[0]
                if dd < min_dist:
                    min_dist = dd
                    NP = temp_root.point

        return NP, min_dist

trainData = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
t = KD_Tree(trainData)
t.buildTree()
t.preorder()
print t.findNearest([2, 4.5])
