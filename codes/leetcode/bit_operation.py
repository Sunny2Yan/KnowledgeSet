"""位运算

| 符号  | 描述  | 规则                              |
| :--: | :--: | :----                            |
| &    |  与  | 两个位都为 1 时，结果才为 1           |\
| \|   |  或  | 有一个位为 0 时，结果就为 0           |\
| ^    | 异或 | 两个位相同为 0，相异为 1              |\
| ~    | 取反 | 0 变 1，1 变 0                     |\
| <<   | 左移 | 全部向左移动 $n$ 位，低位补 0~(\*2^n) |\
| >>   | 右移 | 全部向右移动 $n$ 位，高位补 0，低位去掉 |
"""


class BitOperation(object):
    def __init__(self):
        ...

    @staticmethod
    def hamming_weight(n: int) -> int:
        """二进制中 1 的个数
        (leetcode 191, 剑指offer 15)
        思路：n&(n-1)可以把n的二进制位中的最低位的1变为0，如：6=(110), 4=(100), 6&(6−1)=4，
            因此可以将n一直运算到0即可。
        时O(logn), 空O(1) **
        """
        res = 0
        while n != 0:
            n &= n - 1
            res += 1
        return res

    @staticmethod
    def add(a: int, b: int) -> int:
        """不用加减乘除做加法
        (剑指offer 65) 设两数字的二进制形式a, b，其求和 s=a+b。
        思路：a(i)和b(i)分别表示a和b的二进制第i位，则有以下四种情况:
            |a(i) | b(i) | 无进位和 n(i) | 有进位和 c(i) |
            | --- | ---  | ----------- | ----------- |
            | 0   | 0    |      0      |      0      |\
            | 0   | 1    |      1      |      0      |\
            | 1   | 0    |      1      |      0      |\
            | 1   | 1    |      0      |      1      |
            于是，无进位和n=a^b；有进位和c=a&b<<1。从而s=a+b => s=n+c，循环至c=0即可。
        时O(logn); 空O(1)
        """
        x = 0xffffffff
        a, b = a & x, b & x
        while b != 0:
            a, b = (a ^ b), (a & b) << 1 & x
        return a if a <= 0x7fffffff else ~(a ^ x)

    @staticmethod
    def single_number(nums: list[int]) -> int:
        """只出现一次的数字
        (leetcode 136) 非空整数数组，某个元素只出现一次，其余元素均出现两次，找出该元素。
        思路：由于任何数同自身做异或运算结果都为0（即，a^a=0）。从而将全部数都做异或运算即可。
        时O(n); 空O(1)
        """
        # return sum(set(nums)) * 2 - sum(nums)
        res = 0
        for num in nums:
            res ^= num
        return res
