"""贪心思想属于动态规划思想中的一种，其基本原理是:
找出整体当中给的每个局部子结构的最优解，并且最终将所有的这些局部最优解结合起来形成整体上的一个最优解
"""


class Greedy(object):
    def __init__(self):
        ...

    @staticmethod
    def min_mum_number_of_host(n: int, start_end: list[list[int]]) -> int:
        """主持人调度
        () [[start_1, end_1], ..., [start_n, end_n]], 判断需要几个主持人
        思路：分别对起始区间和终止区间排序，如果第i个起始值大于第j个结束值，则不需要主持人，否则主持人数量加1
        时O(nlogn); 空O(n)
        """
        start, end = list(), list()
        for i in range(n):
            start.append(start_end[i][0])
            end.append(start_end[i][1])
        start.sort()
        end.sort()
        count = 0
        j = 0
        for i in range(n):
            # 新开始的节目大于上一轮结束的时间，主持人不变
            if start[i] >= end[j]:
                j += 1
            else:
                count += 1
        return count

    @staticmethod
    def candy(arr: list[int]) -> int:
        """糖果分配（hard）
        () 根据数组中的分数分配糖果，每个孩子至少一颗，相邻的孩子之间，得分较多的孩子必须拿多一些糖果，返回最少需要多少糖果
        思路：
        时O(n); 空O(n)
        """
        res = [1 for _ in range(len(arr))]
        # 从左往右遍历，递增+1
        for i in range(1, len(arr)):
            if arr[i] > arr[i-1]:
                res[i] = res[i-1] + 1
        count = res[len(arr)-1]  # 记录糖果数
        # 从右往左遍历，递增+1
        for i in range(len(arr)-2, -1, -1):
            if arr[i] > arr[i+1] and res[i] <= res[i+1]:
                res[i] = res[i+1] + 1
            count += res[i]

        return count
