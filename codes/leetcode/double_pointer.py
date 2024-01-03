""""""


class DoublePointer:
    @staticmethod
    def two_sum(nums: list[int], target: int) -> list[int]:
        """和为s的两个数字
        (剑指offer 57) 在递增排序的数组中找出和为target的两个数。
        思路：双指针
        时O(n); 空O(1)
        """
        left, right = 0, len(nums) - 1

        while left < right:
            s = nums[left] + nums[right]
            if s == target:
                return [nums[left], nums[right]]
            elif s < target:
                left += 1
            else:
                right -= 1

        return []

    @staticmethod
    def three_sum(nums: list[int]) -> list[list[int]]:
        """三数之和
        (leetcode 15) 给定数组nums，返回数组中三个不同的元素使其和为0。
        思路：先排序为应用双指针做准备，再遍历确定一个元素，剩余元素应用双指针确定。
        时O(n^2); 空O(1)
        """
        nums.sort()
        res = []

        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:  # 过滤相同的数
                continue

            left, right = i + 1, len(nums) - 1
            while left < right:
                s = nums[i] + nums[left] + nums[right]
                if s == 0:
                    res.append([nums[i], nums[left], nums[right]])
                    while left + 1 < right and nums[left + 1] == nums[left]:
                        left += 1
                    while left < right - 1 and nums[right - 1] == nums[right]:
                        right -= 1
                    left += 1
                    right -= 1

                elif s < 0:
                    left += 1
                else:
                    right -= 1

        return res

    @staticmethod
    def n_sum_sequence(target: int) -> list[list[int]]:
        """和为s的连续正数序列
        (剑指offer 57) 给定正整数target，返回所有和为target的连续正整数序列。
        思路：定义两个指针，且右指针小于target。利用等差数列求和公式S_n=n/2 * (a_1 + a_n)
        时O(n); 空O(1)
        """
        res = []
        left, right = 1, 2

        while left < right < target:
            s = (left + right) * (right - left + 1) / 2
            if s == target:
                res.append(list(range(left, right + 1)))
                left += 1
                right += 1
            elif s < target:
                right += 1
            else:
                left += 1
        return res

    @staticmethod
    def merge_nums(nums1: list[int], m: int,
                   nums2: list[int], n: int) -> list[int]:
        """合并两个有序数组
        (leetcode 88) 将升序数组nums2按升序合并到升序数组nums1（长度为n+m）中。
        思路：分别为两数组定义一个指针，倒序遍历，注意指针左移的规则。
        时O(n); 空O(1)
        """
        left, right = m - 1, n - 1

        for i in range(m + n - 1, -1, -1):
            if right < 0 or (left >= 0 and nums1[left] >= nums2[right]):
                nums1[i] = nums1[left]
                left -= 1
            else:
                nums1[i] = nums2[right]
                right -= 1

        return nums1

    @staticmethod
    def max_area(height: list[int]) -> int:
        """盛最多水的容器
        (leetcode 11) 给定数组height，每个元素表示容器的高度，找出两个高度，使其构成的容器可以容纳最多的水。
        思路：定义左右两个指针，s=min(h[l], h[r])*(r-l)，且高度小的指针移动。
        时O(N); 空O(1)
        """
        if len(height) < 2: return 0
        left, right = 0, len(height) - 1
        res = 0

        while left < right:
            s = min(height[left], height[right]) * (right - left)
            res = max(res, s)
            if height[left] <= height[right]:
                left += 1
            else:
                right -= 1

        return res

    @staticmethod
    def trap(height: list[int]) -> int:
        """接雨水
        (leetcode 42) 给定n个非负整数表示宽度为 1 的柱子的高度，计算下雨之后能接多少雨水。
        思路：只需要维护三个指针（left_max, right_max, cur），可以从两端往中间找，则当前指针
        可分为左、右双指针。其结果为最大值减去当前指针。
        时O(n); 空O(1)
        """
        if len(height) <= 2: return 0
        left, right = 0, len(height) - 1
        max_left, max_right = 0, 0
        res = 0

        while left < right:
            max_left = max(max_left, height[left])
            max_right = max(max_right, height[right])
            if max_left <= max_right:
                res += max_left - height[left]
                left += 1
            else:
                res += max_right - height[right]
                right -= 1

        return res