"""Hash
将数据存储到字典中，使其空间复杂度降低为1.
"""


class HashTable(object):
    @staticmethod
    def two_sum(nums: list[int], target: int) -> list[int]:
        """两数之和
        (leetcode 2) 找出整数数组nums中和为target的两个整数的下标
        思路：遍历列表，如果 target-nums[i] 不在 hash 表中，则加入到哈希表中，否则返回索引。
        时O(n); 空O(n)"""
        hash_table = {}

        for i in range(len(nums)):
            if target - nums[i] in hash_table:
                return [i, hash_table[target - nums[i]]]
            else:
                hash_table[nums[i]] = i

        return []

    @staticmethod
    def unique_occurrences(nums: list[int]) -> int:
        """独一无二的出现次数
        (leetcode 1207) 整数数组nums，如果数组中每个数的出现次数都是独一无二的，就返回 true，否则返回 false
        思路：建立hash表对数组中的数进行计数，最后遍历hash表即可。
        时O(n); 空O(1)
        """
        hash_map = dict()
        for num in nums:
            if num not in hash_map:
                hash_map[num] = 1
            else:
                hash_map[num] += 1

        my_set = set()
        for i in hash_map:
            my_set.add(hash_map[i])

        return len(my_set) == len(hash_map)