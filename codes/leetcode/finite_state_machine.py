# -*- coding: utf-8 -*-

class AutoMaton:
    INT_MIN, INT_MAX = -2 ** 31, 2 ** 31 - 1

    def __init__(self):
        ...

    def atoi(self, s: str):
        """字符串转换整数
        (leetcode 8) 丢弃前导空格，检查正负号，直到读入的下一个非数字字符或到达输入的结尾。整数范围 [−2^31,  2^31−1]。
        思路：设每个时刻的状态为 S，每次输入一个字符 C，根据当前状态 S 和输入字符 C 转移到下一个状态 S‘，即：S,C -> S’。
        时O(n) 空O(1)
        """
        state, sign, res = 'start', 1, 0
        table = {
            'start': ['start', 'signed', 'in_number', 'end'],
            'signed': ['end', 'end', 'in_number', 'end'],
            'in_number': ['end', 'end', 'in_number', 'end'],
            'end': ['end', 'end', 'end', 'end'], }

        for c in s:
            if c.isspace():
                state = table[state][0]
            elif c == '+' or c == '-':
                state = table[state][1]
            elif c.isdigit():
                state = table[state][2]
            else:
                state = table[state][3]

            if state == 'in_number':
                res = res * 10 + int(c)
                res = min(res, self.INT_MAX) if sign == 1 else min(
                    res, -self.INT_MIN)
            elif state == 'signed':
                sign = 1 if c == '+' else -1

        return sign * res


if __name__ == '__main__':
    automaton = AutoMaton()
    s = "42"
    print(automaton.atoi(s))