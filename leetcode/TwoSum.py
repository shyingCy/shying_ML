class Solution(object):

    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        if len(nums) <= 1:
            return []

        nums_dict = {}
        for i in range(len(nums)):
            if nums[i] in nums_dict.keys():
                return [nums_dict[nums[i]], i]
            else:
                nums_dict[target - nums[i]] = i

s = Solution()
print s.twoSum([-3, 4, 3, 90], 0)
