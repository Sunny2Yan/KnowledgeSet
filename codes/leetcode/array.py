from typing import Optional


class FindArray(object):
    @staticmethod
    def find_number(nums: list[int], target: int) -> int:
        """查找有序列表中目标数字重复次数
        （剑指offer 53）二分法，定义一个函数 $f$ 来找到第一个大于target的索引，然后根据
        $f(target) - f(target - 1)$ 即可。 [1,2,3,3,4,4,5]
        时O(logn), 空O(1)
        """
        def __find(tar):
            left, right = 0, len(nums)-1
            while left <= right:
                mid = left + (right - left) // 2
                if nums[mid] <= tar:
                    left = mid + 1
                else:
                    right = mid - 1
            return left

        return __find(target) - __find(target - 1)

    @staticmethod
    def find_renumber(nums: list[int]) -> Optional[int]:
        """查找数组中重复的数字
        （剑指offer 03）建立集合存储数字，遍历数组，集合中不存在就加入，否则输出。
        时O(n)，空O(1)
        """
        dict = set()
        for num in nums:
            if num not in dict:
                dict.add(num)
            else:
                return num

        return None

    @staticmethod
    def find_loss_number(nums: list[int]) -> int:
        """查找 0~n-1 中缺失的数字
        （剑指offer 53）二分法，若中间数字的值等于索引，则右侧缺失，否则左侧缺失，由于每次先判断
        left，最终只需返回left即可。也可以前n项和直接算。 [0,1,2,3,4,6,7,8,9]
        时O(logn), 空O(1)
        """
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == mid:
                left = mid + 1
            else:
                right = mid - 1
        return left

    @staticmethod
    def find_matrix(matrix: list[list[int]], target: int) -> bool:
        """递增矩阵查找元素: 矩阵中行递增，列递增，判断矩阵中是否含target值
        （剑指offer 04）从左下角查起，如果第一个元素小于target，查找这一行的右侧，否则查找上一行。
        时O(m + n)，空O(1)
        """
        if not matrix: return False
        left, right = len(matrix) - 1, 0  # 从左下角查起
        while left >= 0 and right < len(matrix[0]):
            if matrix[left][right] < target:
                right += 1
            elif matrix[left][right] > target:
                left -= 1
            else:
                return True
        return False


class TransposeArray(object):
    @staticmethod
    def transpose_matrix(matrix: list[list[int]]) -> list[list[int]]:
        """旋转矩阵（转置）
        （leetcode 867）定义一个相同大小的矩阵，然后进行值替换。
        时O(nm), 空O(mn)
        """
        # 1. list(zip(*matrix))
        # 2. np.array(matrix).T.tolist()
        new_matrix = [[0] * len(matrix) for _ in matrix[0]]
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                new_matrix[j][i] = matrix[i][j]

        return new_matrix

    @staticmethod
    def print_matrix(matrix: list[list[int]]) -> list:
        """顺时针打印矩阵
        (剑指offer 29) 如果 $left <= right or up <= down$ 顺时针打印，先打印顶横和右竖，
        再打印下横和左竖，但$left \neq right and up\neq down$。
        O(n^2)；空O(n^2)
        """
        res = []
        if not matrix or not matrix[0]: return res
        left, up, right, down = 0, 0, len(matrix[0]) - 1, len(matrix) - 1
        while left <= right or up <= down:  # 顶部横，右边竖
            for row in range(left, right + 1):
                res.append(matrix[up][row])
            for col in range(up + 1, down + 1):
                res.append(matrix[col][right])
            if left < right and up < down:  # 底部横，左边竖
                for row in range(right - 1, left, -1):
                    res.append(matrix[down][row])
                for col in range(down, up, -1):
                    res.append(matrix[col][left])
            left, up, right, down = left + 1, up + 1, right - 1, down - 1

        return res


class ComputeArray(object):
    @staticmethod
    def multiply_matrix(matrix_1: list[list[int]],
                        matrix_2: list[list[int]]) -> list[list[int]]:
        """矩阵乘法运算"""
        res = [[] for _ in range(len(matrix_1))]
        for i in range(len(matrix_1)):
            for j in range(len(matrix_2[0])):
                _sum = 0
                for k in range(len(matrix_1[0])):
                    _sum += matrix_1[i][k] * matrix_2[k][j]
                res[i].append(_sum)

        return res

    @staticmethod
    def merge_arr(intervals: list[list[int]]) -> list[list[int]]:
        """合并区间
        (leetcode 56) intervals包含若干个区间的数组，合并所有重叠的区间，返回一个不重叠的区间数组。
        思路：先将数组按第一维度排序，再比较前一个区间的end与下一个区间的start的大小
        时O(nlogn); 空O(1)"""
        intervals.sort(key=lambda x: x[0])
        res = []

        for interval in intervals:
            if not res or res[-1][1] < interval[0]:
                res.append(interval)
            else:
                res[-1][1] = max(res[-1][1], interval[1])

        return res

    @staticmethod
    def spiral_order(nums: list[int]) -> list[list[int]]:
        """三数之和
        (leetcode 15) 给定数组nums，返回数组中三个不同的元素使其和为0。
        思路：先排序为应用双指针做准备，再遍历确定一个元素，剩余元素应用双指针确定。
        时O(n^2); 空O(1)
        """
        nums.sort()
        res = []

        for i in range(len(nums)-2):
            if i > 0 and nums[i] == nums[i-1]:  # 过滤相同的数
                continue

            left, right = i+1, len(nums)-1
            while left < right:
                s = nums[i] + nums[left] + nums[right]
                if s == 0:
                    res.append([nums[i], nums[left], nums[right]])
                    while left+1 < right and nums[left+1] == nums[left]:
                        left += 1
                    while left < right-1 and nums[right-1] == nums[right]:
                        right -= 1
                    left += 1
                    right -= 1

                elif s < 0:
                    left += 1
                else:
                    right -= 1

        return res

    @staticmethod
    def generate(num_rows: int) -> list[list[int]]:
        """杨辉三角
        (leetcode 118) 给定非负整数num_rows，生成杨辉三角的前num_rows行.
        思路：可以一行一行地计算杨辉三角，每当计算出第i行的值，就可以求出第i+1行的值。
        时O(n^2); 空O(n!)
        """
        res = []

        for i in range(num_rows):
            temp = []
            for j in range(i+1):
                if j == 0 or j == i:  # 头尾两个值都是1
                    temp.append(1)
                else:                 # 中间值需要前一行的值求和
                    temp.append(res[i-1][j-1] + res[i-1][j])
            res.append(temp)

        return res


if __name__ == '__main__':
    find_array = FindArray()
    transpose_array = TransposeArray()
    compute_array = ComputeArray()