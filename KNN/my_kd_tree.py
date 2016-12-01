# -*-coding:utf-8-*-
_author_ = "shying"

class KD_node:
    root = []
    data = []
    def __int__(self, point=None, split=None, LL=None, RR=None):
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
    def fit(self, train_data):
        data = train_data


def buildTree(root, data):

    if len(data) == 0:
        return

    features_num = len(data[0])

