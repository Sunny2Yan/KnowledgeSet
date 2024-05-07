"""找递推公式"""


class FstDP(object):
    def __init__(self):
        pass

    @staticmethod
    def climb_stairs(n: int) -> int:
        """爬楼梯
        (leetcode 70) 每次可以爬1，2阶，有多少种爬法。
        思路：f(n)=f(n-1) + f(n-2), 递归 2^n。
        时O(n) 空O(1)
        """
        dp = [0, 0, 1]
        for i in range(n):
            dp[0] = dp[1]
            dp[1] = dp[2]
            dp[2] = dp[0] + dp[1]

        return dp[2]

    @staticmethod
    def max_sub_array(nums: list[int]) -> int:
        """连续子数组的最大和
        (leetcode 53; 剑指offer 42) 找最大和的连续子数组，并返回最大和。
        思路：f(n)=max(f(n-1)+f(n), f(n))
        时O(n) 空O(n)
        """
        dp = [num for num in nums]
        for i in range(1, len(nums)):
            dp[i] = max(dp[i-1]+nums[i], nums[i])

        return max(dp)

    @staticmethod
    def rob(nums: list[int]) -> int:
        """打家劫舍
        (leetcode 198) 求元组不取相邻位置值的最大和。
        思路：f(n)=max(f(n-1) f(n)+f(n-2))。
        时O(n) 空O(n)
        """
        if not nums:
            return 0
        elif len(nums) == 1:
            return nums[0]
        dp = [num for num in nums]
        dp[1] = max(dp[0], dp[1])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])

        return dp[-1]

    @staticmethod
    def max_profit_1(prices: list[int]) -> int:
        """买卖股票的最佳时机1
        (leetcode 121; 剑指offer 63) 某天买入这支股票，在未来卖出，求获得利润最大（只买卖一次）。
        思路：f(n)=max(max_profit, price-min_price)
        """
        min_price = prices[0]
        max_profit = 0

        for price in prices:
            min_price = min(min_price, price)
            max_profit = max(max_profit, price-min_price)

        return max_profit

    @staticmethod
    def max_profit_2(prices: list[int]) -> int:
        """买卖股票的最佳时机2
        (leetcode 122) 每一天都可以购买或卖出，但最多可以持有一股。
        思路：定义状态dp[i][0]表示第$i$天交易完后手里没有股票的最大利润；
            dp[i][1]表示第i天交易完后手里持有一支股票的最大利润。
            则 dp[i][0]=max{dp[i-1][0], dp[i-1][1]+price[i]}，
            dp[i][1]=max{dp[i-1][1], dp[i-1][0]-price[i]}，
            则结束后一定是手上没有股票的收益最大。
        时O(n); 空O(1)
        """
        if len(prices) <= 1: return 0

        dp_i_0, dp_i_1 = 0, -prices[0]
        for i in range(1, len(prices)):
            dp_i_0 = max(dp_i_0, dp_i_1 + prices[i])
            dp_i_1 = max(dp_i_1, dp_i_0 - prices[i])

        return dp_i_0

    @staticmethod
    def nth_ugly_number(n: int) -> int:
        """丑数
        (leetcode 241; 剑指offer 49) 包含质因子2、3、5的数称为丑数，求从小到大的第n个丑数
        思路：dp[i]=min(dp[a]*2, dp[b]*3, dp[c]*5)
        """
        dp = [1] * n
        a, b, c = 0, 0, 0
        for i in range(1, n):
            dp[i] = min(dp[a]*2, dp[b]*3, dp[c]*5)
            if dp[i] == dp[a] * 2: a += 1
            if dp[i] == dp[b] * 3: b += 1
            if dp[i] == dp[c] * 5: c += 1

        return dp[-1]


class SecDP(object):
    def __init__(self):
        ...

    @staticmethod
    def min_path_sum_1(grid: list[list[int]]) -> int:
        """最小/大路径和
        (leetcode 64; 剑指offer 47) mxn的网格grid，找一条从左上角到右下角的路径，使路径和最小。
        思路：（同最大路径和）f(0, 0) = f(0, 0)
            f(0, j) = grid(0, j) + grid(0, j-1);
            f(i, 0) = grid(i, 0) + grid(i-1, 0);
            f(i, j) = grid(i, j) + min(grid(i, j-1), grid(i-1, j))
        时O(mn); 空O(mn)
        """
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if i == 0 and j == 0:
                    continue
                elif i == 0:
                    grid[0][j] = grid[0][j] + grid[0][j-1]
                elif j == 0:
                    grid[i][0] = grid[i][0] + grid[i-1][0]
                else:
                    grid[i][j] = grid[i][j] + min(grid[i-1][j], grid[i][j-1])
        return grid[-1][-1]

    @staticmethod
    def min_path_sum_2(triangle: list[list[int]]) -> int:
        """三角形最小路径和
        (leetcode 120) 给定三角形，找出自顶向下的最小路径和。
        思路:f(i, 0) = f(i-1, 0) + t(i, 0)  左边缘
            f(i, i) = f(i-1, i-1) + t(i, i)  右边缘
            f(i, j) = min(f(i-1, j-1), f(i-1, j)) + t(i, j)
        时O(n^2); 空O(n)
        """
        dp = [[triangle[0][0]] * i for i in range(1, len(triangle) + 1)]
        for i in range(1, len(triangle)):
            dp[i][0] = dp[i - 1][0] + triangle[i][0]  # 左边缘
            for j in range(1, i):
                dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle[i][j]
            dp[i][i] = dp[i - 1][i - 1] + triangle[i][i]  # 右边缘

        return min(dp[-1])

    @staticmethod
    def knapsack_1(C: int, V: list[int], W: list[int]) -> int:
        """01背包问题
        () 已知背包的体积C，物品的体积V=[v_i]和价值W=[w_i]，求背包最大能装多大价值的物品
        思路：定义i表示第i件物品，j为背包的剩余空间，则剩余空间大于第i个物品体积时选择拿取有：
            dp[i][j] = max(dp[i-1][j], dp[i-1][j-V[i]) + W[i]；
            当剩余空间小于第i个物品体积时，只能不拿：dp[i][j] = dp[i-1][j]
        时O(mn); 空O(mn)
        """
        dp = [[0 for _ in range(C+1)] for _ in range(len(V) + 1)]
        for i in range(1, len(V)+1):
            for j in range(1, C+1):
                if j >= V[i-1]:
                    dp[i][j] = max(dp[i-1][j], dp[i-1][j-V[i-1]] + W[i-1])
                else:
                    dp[i][j] = dp[i-1][j]

        return dp[-1][-1]

    @staticmethod
    def knapsack_2(C: int, V: list[int], W: list[int]) -> int:
        """完全背包问题
        () 已知背包的体积C，物品的体积V=[v_i]和价值W=[w_i]，每种物品任意多个，求背包最大能装多大价值的物品.
        思路：剩余空间小于第i-1个物品体积时，只能不拿：dp[i][j] = dp[i-1][j];
            剩余空间大于等于第i-1个物品时：dp[i][j] = max(dp[i-1][j], dp[i][j-V[i]] + W[i])
        时O(mn); 空O(mn)
        """
        dp = [[0 for _ in range(C+1)] for _ in range(len(V)+1)]
        for i in range(1, len(V)+1):
            for j in range(1, C+1):
                if j >= V[i-1]:
                    #
                    dp[i][j] = max(dp[i-1][j], dp[i][j-V[i-1]] + W[i-1])
                else:
                    dp[i][j] = dp[i-1][j]

        return dp[-1][-1]

    @staticmethod
    def find_length(nums1: list[int], nums2: list[int]) -> int:
        """最长重复子数组
        (leetcode 718) 给两个整数数组，返回两个数组中公共的、长度最长的子数组长度。
        思路：令dp[i][j]表示nums1[i:]和nums2[j:]的最长公共前缀，则dp的最大值就是结果。
            若nums1[i] == nums2[j]则dp[i][j] = dp[i-1][j-1] + 1;
            否则dp[i][j]=0
        时O(n^2); 空O(n^2)
        """
        dp = [[0 for _ in range(len(nums2) + 1)] for _ in range(len(nums1) + 1)]
        res = 0

        for i in range(1, len(nums1) + 1):
            for j in range(1, len(nums2) + 1):
                if nums1[i - 1] == nums2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    res = max(res, dp[i][j])

        return res

    @staticmethod
    def longest_palindrome(s: str) -> str:
        """最长回文子串
        (leetcode 5) 给一个字符串，找出字符串中最长的回文子串。
        思路：对于一个长度大于2的子串，首位字母去掉后仍是回文串。令dp[i][j]表示s[i:j+1]的子串，
            dp[i][j]=(dp[i+1][j-1] and s[i] == s[j])
        时O(n^2); 空O(n^2)
        """
        if len(s) <= 1: return s

        dp = [[False for _ in range(len(s))] for _ in range(len(s))]
        res = ""

        for length in range(len(s)):  # 字符串长度
            for left in range(len(s)):  # 左边界
                right = left + length  # 右边界
                if right >= len(s): break

                if length == 0 or length == 1:
                    dp[left][right] = (s[left] == s[right])
                else:
                    dp[left][right] = (dp[left+1][right-1] and s[left] == s[right])

                if dp[left][right] and length+1 > len(res):
                    res = s[left:right+1]
        return res


if __name__ == '__main__':
    fstdp = FstDP()
    secdp = SecDP()

    print(fstdp.climb_stairs(10))

    print(secdp.knapsack_2(10, [1, 9], [3, 8]))


