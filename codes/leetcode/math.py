"""数学问题

"""


class Math(object):
    def __init__(self):
        ...

    @staticmethod
    def reverse(x: int) -> int:
        """整数反转
        (leetcode 7) 给定一个有符号整数x，返回将x中的数字反转后的结果，若结果超出[−2^31, 2^31 − 1]，则返回0
        思路：将x每次整除10，且记录余数。注意x的符号
        时O(logn); 空O(1)
        """
        flag = 1 if x >= 0 else -1
        res, y = 0, abs(x)
        while y > 0:
            res = res * 10 + y % 10
            if res * flag < -2**31 or res * flag > 2**31-1:
                return 0
            y //= 10

        return res * flag

    @staticmethod
    def greatest_common_divisor(x: int, y: int) -> int:
        """求最大公约数
        """
        for factor in range(min(x, y), 0, -1):
            if x % factor == 0 and y % factor == 0:
                return factor

    @staticmethod
    def min_common_multiple(x: int, y: int) -> int:
        """求最小公倍数
        """
        res = 1
        for i in range(1, min(x, y)):
            if x % i == 0 and y % i == 0:
                res *= i
                x //= i
                y //= i

        return x * y * res

    @staticmethod
    def is_prime(n: int) -> bool:
        """判断是不是素数
        """
        for factor in range(2, int(n ** 0.5) + 1):
            if n % factor == 0:
                return False
        return True if n != 1 else False

    @staticmethod
    def cube_root(n: int) -> int:
        """求立方根
        思路：牛顿迭代法求解。f(x)=x^3，由f(x_1) = f(x_0) + f'(x_0)(x_1 - x_0)可求 x_1
        """
        res, e = n, 0.01
        while abs(res**3 - n) > e:
            res -= (res ** 3 - n) / (3 * res ** 2)

        return res

    @staticmethod
    def cutting_rope(n: int) -> int:
        """剪绳子
        (剑指offer 12) 给定一条长为n的绳子，将绳子剪成整数长度的m段（n>1 and m>1），计算m段的长度乘积
        思路：argmax(n1*...*n_m) s.t. n=n_1+...+n_m;由算术几何均值不等式(n1+...+n_m)/m >= pow(n1*...*n_m, m) 当且仅当 n1=...=n_m时等号成立，从而每段长度相等时最大；
        """
        if n <= 3:
            return n - 1
        a, b = n // 3, n % 3
        if b == 1:
            return int(3 ** (a - 1) * 4)
        if b == 2:
            return int(3 ** a * 2)

        return int(3 ** a)

    @staticmethod
    def joseph_circle(n: int, m: int) -> int:
        """约瑟夫环
        (剑指offer 62) 给定整数n，将0...n-1围成一个圆，从数字0开始，每次从圆圈里面删除第m个数字，求这个圆圈里剩下的最后一个数字
        思路：每次将nums[:m]删除，并添加到nums末尾，组成新的数组，要删除的数字在新的nums最后一位
        """
        nums, num = [i for i in range(n)], 0
        while nums:
            for _ in range(m):
                temp = nums.pop(0)
                nums.append(temp)

            num = nums.pop()

        # nums, idx = [i for i in range(n)], 0
        # for i in range(n, 0, -1):
        #     idx = (idx + m - 1) % i  # 本轮第m个位置
        #     num = nums.pop(idx)
        return num



graph = [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 0, 1, 1, 1, 1, 0, 0, 0],
         [1, 1, 1, 0, 1, 1, 0, 1, 0, 0],
         [1, 1, 1, 1, 0, 1, 0, 0, 1, 0],
         [1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
         [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
         [1, 1, 0, 1, 0, 0, 1, 0, 1, 1],
         [1, 1, 0, 0, 1, 0, 1, 1, 0, 1],
         [1, 1, 0, 0, 0, 1, 1, 1, 1, 0]]

import numpy as np

graph = np.array(graph)
graph_cube = graph @ graph @ graph
print(graph_cube)
res = graph_cube.trace() / 6 - 24  # 3 (三角形每个顶点都算一次) * 2 (方向)
print(res)


A = {"A": {"B", "C", "D", "E", "F", "G", "H", "I", "J"},
     "B": {"A", "C", "D", "E", "F", "G", "H", "I", "J"},
     "C": {"A", "B", "D", "E", "F", "G"},
     "D": {"A", "B", "C", "E", "F", "H"},
     "E": {"A", "B", "C", "D", "F", "I"},
     "F": {"A", "B", "C", "D", "E", "J"},
     "G": {"A", "B", "C", "H", "I", "J"},
     "H": {"A", "B", "D", "G", "I", "J"},
     "I": {"A", "B", "E", "G", "H", "J"},
     "J": {"A", "B", "F", "G", "H", "I"}}
B = [["A", "H", "D"], ["A", "I", "E"], ["A", "J", "F"], ["A", "G", "C"],
     ["B", "H", "I", "J", "G"], ["B", "D", "E", "F", "C"]]

res = 0
for k, v in A.items():  # 第一个点
    for i in v:  # 第二个点
        # print(k, i)  # 第一条边
        for j in A[i]:  # 第三个点
            if k == j:  # 第一个点与第三个点重合
                pass
            else:
                if k in A[j]:  # 第一个点是第三个点指向的点
                    res += 1


def factorial(n):
    ans = 1
    for i in range(1, n+1):
        ans *= i
    return ans


def c_n_m(n, m):
     return factorial(n) / (factorial(m) * factorial(n-m))


res = res / 6
for b in B:
    res -= c_n_m(len(b), 3)

print(res)

