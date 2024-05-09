"""先判断终止条件，再递归，最后回溯"""


class Recursion(object):
    @staticmethod
    def permute_1(nums: list[int]) -> list[list[int]]:
        """全排列（没有重复元素）
        (leetcode 46，剑指offer 83) 给一个不含重复数字的数组，返回其所有可能的全排列。
        思路： 定义递归函数，用first表示已经填了的位置，在递归函数中first指向起始位置，递归时，如若first等于列表长度，
        则已经结束，否则first的值与i值交换；然后再递归first+1，注意递归完成后需要回溯，变成原始字符。
        时O(n*n!); 空O(n)
        """
        res = []

        def recursion_fn(first=0):
            if first == len(nums):  # 终止条件
                res.append(nums[:])
            else:
                for i in range(first, len(nums)):
                    nums[first], nums[i] = nums[i], nums[first]  # 动态维护数组
                    recursion_fn(first + 1)
                    nums[first], nums[i] = nums[i], nums[first]

        recursion_fn()
        return res

    @staticmethod
    def permute_2(nums: list[int]) -> list[list[int]]:
        """全排列（包含重复元素）
        (leetcode 47, 剑指offer 84) 给定一个数组，按任意顺序返回所有不重复的全排列。
        思路：基于无重复元素全排列，终止条件添加判断新的 nums[:]不在原结果列表中。
        时O(n*n!); 空O(n)
        """
        res = []

        def recursive(first=0):
            if first == len(nums) and nums[:] not in res:
                res.append(nums[:])
            else:
                for i in range(first, len(nums)):
                    nums[first], nums[i] = nums[i], nums[first]
                    recursive(first+1)
                    nums[first], nums[i] = nums[i], nums[first]

        recursive()
        return res

    @staticmethod
    def combine(n: int, k: int) -> list[list[int]]:
        """组合
        (leetcode 77) 返回数组[1, n]中所有可能的 k 个数的组合。
        思路：同全排列，并将i放入temp，再递归 i+1
        时O(C_n^k * k); 空O(n)
        """
        res, temp = [], []

        def recursive(first=1):
            if len(temp) == k:
                res.append(temp[:])
            else:
                for i in range(first, n + 1):
                    temp.append(i)
                    recursive(i + 1)
                    temp.pop()

        recursive()
        return res

    @staticmethod
    def combination_sum_1(candidates: list[int], target: int) -> list[list[int]]:
        """组合总和（无重复元素）
        (leetcode 39) 给定无重复元素的整数数组candidates和整数target，找出candidates 中和为target的所有不同组合。
        思路：整体同上，但由于同一个数字可以被多次使用，所以递归时，每次要从起点开始。
        时O(n * 2^n); 空O(n)
        """
        res, temp = [], []

        def recursive(first=0, remain=target):
            if target == 0:
                res.append(temp[:])
            elif target < 0:
                return None
            else:
                for i in range(first, len(candidates)):
                    temp.append(candidates[i])
                    recursive(i, remain-candidates[i])  # 可以重复使用
                    temp.pop()

        recursive()
        return res

    @staticmethod
    def combination_sum_2(candidates: list[int], target: int) -> list[list[int]]:
        """组合总和（包含重复元素）
        (leetcode 40) 给定含重复元素数组candidates和目标数target，找出candidates中和为target的组合。
        思路：含有重复元素，需要去重，对candidates排序并判断temp[:]不在res中，注意此题每个数字只使用一次。
        时O(n * 2^n); 空O(n)
        """
        res, temp = [], []
        candidates.sort()

        def recursive(first=0, remain=target):
            if target == 0 and temp[:] not in res:
                res.append(temp[:])
            elif target < 0:
                return None
            else:
                for i in range(first, len(candidates)):
                    if candidates[i] > target:  # 剪枝，减小复杂度
                        break
                    if i > first and candidates[i-1] == candidates[i]:  # 过滤重复元素
                        continue
                    temp.append(candidates[i])
                    recursive(i+1, remain-candidates[i])  # 每个数字只使用一次
                    temp.pop()

        recursive()
        return res

    @staticmethod
    def subsets_1(nums: list[int]) -> list[list[int]]:
        """子集（无重复元素）
        (leetcode 78, 剑指offer 79) 给定数组nums，返回数组所有的子集。
        思路：将每一个temp都加入到res中，无需终止条件。
        时O(n * 2^n); 空O(n)
        """
        res, temp = [], []

        def recursive(first=0):
            res.append(temp[:])  # 第一次为 []
            for i in range(first, len(nums)):
                temp.append(nums[i])
                recursive(i + 1)
                temp.pop()

        recursive()
        return res

    def subsets_2(self, nums: list[int]) -> list[list[int]]:
        """子集（包含重复元素）
        (leetcode 90) 给定含重复元素的数组nums，返回数组所有的子集。
        思路：同上，但由于含有重复元素，需要去重，对nums排序并判断temp[:]不在res中
        时O(n * 2^n); 空O(n)
        """
        res, temp = [], []
        nums.sort()

        def recursive(first=0):
            if temp[:] not in res:
                res.append(temp[:])

            for i in range(first, len(nums)):
                temp.append(nums[i])
                recursive(i + 1)
                temp.pop()

        recursive()
        return res
