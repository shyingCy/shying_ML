class Solution(object):
    def swap(self, a, b):
        c = a
        a = b
        b = c
        return a, b

    def sort(self, nums):
        ind = range(len(nums))
        for ii in range(len(nums) - 1):
            for j in range(len(nums) - 1 - ii):
                if nums[j] > nums[j+1]:
                    nums[j], nums[j+1] = self.swap(nums[j], nums[j+1])
                    ind[j], ind[j+1] = self.swap(ind[j], ind[j+1])
        return nums, ind

    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        nums, ind = self.sort(nums)
        nums_order = {}
        sml_ind = 0
        for i in range(len(nums)):
            if target >= nums[i]:
                sml_ind = i

            if nums[i] not in nums_order.keys():
                nums_order[nums[i]] = []
            nums_order[nums[i]].append(ind[i])

        for i in range(len(nums)):
            exce_num = target - nums[i]
            if exce_num in nums_order.keys():
                if exce_num != nums[i]:
                    return [ind[i], nums_order[exce_num][0]]
                else:
                    for orde in nums_order[exce_num]:
                        if orde != ind[i]:
                            return [ind[i], orde]

        return []

s = Solution()
print s.twoSum([-3, 4, 3, 90], 0)
