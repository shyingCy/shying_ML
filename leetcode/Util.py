# -*-coding:utf-8 -*-
'''
Created on Feb 23, 2017

@author: Shying
'''
def swap(self, a, b):
    c = a
    a = b
    b = c
    return a, b


def sort(self, nums):
    ind = range(len(nums))
    for ii in range(len(nums) - 1):
        for j in range(len(nums) - 1 - ii):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = self.swap(nums[j], nums[j + 1])
                ind[j], ind[j + 1] = self.swap(ind[j], ind[j + 1])
    return nums, ind
