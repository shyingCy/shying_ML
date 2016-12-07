# -*-coding:utf-8-*-

"""
测试用牛顿法求数的平凡根
"""
def newton_sqrt(x):
    k = 1.0
    while abs(k ** 2 - x) > 1e-10:
        k = 0.5 * (k + x/k)
    return k

print newton_sqrt(2)
