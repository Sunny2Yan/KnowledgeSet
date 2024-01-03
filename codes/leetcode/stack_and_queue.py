"""
stack: 先进后出；
queue: 先进先出。
"""
import collections


# 1.stack
class Stack(object):
    @staticmethod
    def is_valid(s: str) -> bool:
        """有效的括号
        (leetcode 20) 给定只包括'('，')'，'{'，'}'，'['，']'的字符串 s，判断是否有效.
        时O(n); 空O(n)
        """
        stack, hash_map = [], {')': '(', '}': '{', ']': '['}

        for i in s:
            if i not in hash_map:
                stack.append(i)
            else:
                if not stack or stack[-1] != hash_map[i]:
                    return False
                stack.pop()

        return not stack  # 最终的stack为空，则返回对

    @staticmethod
    def generate_parenthesis(n: int) -> list[str]:
        """括号生成
        (leetcode 22) n为生成括号的对数，返回有效的括号组合。
        思路：定义栈存储生成的括号s、左括号个数l和有括号个数r，l<n时，s+'('且l+1；r<n且r<l时，s+')'且r+（同样的思想可以改写为DFS，具体见DFS）。
        时O(); 空O()
        """
        stack, res = [('', 0, 0)], []
        while stack:
            s, left, right = stack.pop()
            if left == right == n:
                res.append(s)
            if left < n:
                stack.append((s+'(', left+1, right))
            if right < n and right < left:
                stack.append((s+')', left, right+1))
        return res

    @staticmethod
    def eval_rpn(tokens: list[str]) -> int:
        """逆波兰表达式求值（后缀表达式）
        (leetcode 150) 给定字符串数组tokens，根据逆波兰表示法表示的算术表达式求值。["2","1","+","3","*"] -> (2+1)*3 = 9
        思路：遇到数字就压栈，遇到运算符就弹出栈顶两个数，后一个对前一个做运算，并将计算结果压入栈中，直到栈中只剩下最后一个数。
        时O(n); 空O(n)
        """
        stack = []
        operator2function = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: int(x / y)  # 不能整除 2//-3=-1
        }

        for token in tokens:
            if token not in operator2function:
                stack.append(int(token))
            else:
                num_1 = stack.pop()
                num_2 = stack.pop()
                stack.append(operator2function[token](num_2, num_1))

        return stack[0]

    @staticmethod
    def calculate_1(s: str) -> int:
        """基本计算器II（中缀表达式，无括号）
        (leetcode 227) 给定一个字符串表达式s，实现+-*/并返回计算结果。"3+2*2" -> 7
        思路：先算乘除，如果符号是*/，则从栈内取出一个元素运算，并将结果入栈，否则直接入栈，最后计算栈内元素和。
        时O(n); 空O(n)
        """
        stack, num, pre_operator = [], 0, '+'
        s += "+"  # 每次需要运行到下一个符号，需要添加结束符

        for i in s:
            if i == " ":
                continue
            elif i.isdigit():
                num = num * 10 + int(i)
            else:
                match pre_operator:
                    case "+":
                        stack.append(num)
                    case '-':
                        stack.append(-num)
                    case "*":
                        stack.append(stack.pop() * num)
                    case '/':
                        stack.append(int(stack.pop() / num))  # 不能//，避免负数出现意外

                num, pre_operator = 0, i
        return sum(stack)

    def calculate_2(self, s: str) -> int:
        """基本计算器（中缀表达式，有括号 hard）
        (leetcode 224) 给定字符串表达式s，实现基本计算器来计算并返回结果。"(1+(4+5+2)-3)+(6+8)" -> 23
        思路：
        时O(n); 空O(n)
        """
        s += '+'        # 添加结束符，结束最后一个数字的扫描
        stack = []
        num = 0         # 记录符号之间的数字
        operator = '+'
        idx = 0         # 全局指针
        while idx < len(s):
            if s[idx] == ' ':
                idx += 1
            # 遇到左括号
            if s[idx] == '(':
                temp = idx + 1  # 扫描括号内的局部指针
                lens = 1        # 平衡左右括号，对其后结束循环
                while lens > 0:
                    if s[temp] == '(':
                        lens += 1
                    if s[temp] == ')':
                        lens -= 1
                    temp += 1
                # 将括号视为子问题进入递归
                num = self.calculate_2(s[idx+1: temp-1])
                idx = temp - 1
            # 运算
            if s[idx].isdigit():
                num = num * 10 + int(s[idx])
            else:
                if operator == '+':
                    stack.append(num)
                elif operator == '-':
                    stack.append(- num)
                elif operator == '*':
                    stack.append(stack.pop() * num)
                num = 0
                operator = s[idx]
            idx += 1
        return sum(stack)


# 2.queue
class Queue(object):
    @staticmethod
    def max_sliding_window(nums: list[int], k: int) -> list[int]:
        """滑动窗口的最大值（hard）
        (leetcode 239) 给定整数数组nums，大小为k的窗口从左向右每次滑动一格，返回滑动窗口中的最大值。
        思路：将queue中小于当前元素的元素删除获得单调队列，注意queue[0]==nums[j-k]表示右移时，最左边的元素是否在queue中。
        时O(n); 空O(k)"""

        queue, res = collections.deque(), []

        for i in range(k):
            while queue and queue[-1] < nums[i]:  # 确保单调递减队列
                queue.pop()
            queue.append(nums[i])
        res.append(queue[0])

        for j in range(k, len(nums)):
            if queue[0] == nums[j - k]:  # nums[j-k]在queue中需要删除
                queue.popleft()
            while queue and queue[-1] < nums[j]:
                queue.pop()
            queue.append(nums[j])
            res.append(queue[0])

        return res


# 3.Stack and queue conversion
# 3.1 双栈实现队列
class MyQueue:
    def __init__(self):
        """用栈实现队列
        (leetcode 232) 仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作（push、pop、peek、empty）.
        思路：stack1为主栈，stack2为临时栈
        """
        self.stack1 = []
        self.stack2 = []

    def push(self, x: int) -> None:
        while self.stack1:
            self.stack2.append(self.stack1.pop())
        self.stack1.append(x)
        while self.stack2:
            self.stack1.append(self.stack2.pop())

    def pop(self) -> int:
        return self.stack1.pop()

    def peek(self) -> int:
        return self.stack1[-1]

    def empty(self) -> bool:
        return not self.stack1


# 3.2 双队列实现栈
class MyStack:
    def __init__(self):
        """用队列实现栈
        (leetcode 225) 仅使用两个队列实现一个后入先出的栈，并支持普通栈的全部四种操作（push、top、pop和empty）
        思路：queue1为主队列， queue2为辅助队列
        """
        self.queue1 = collections.deque()
        self.queue2 = collections.deque()

    def push(self, x: int) -> None:
        self.queue2.append(x)
        while self.queue1:
            self.queue2.append(self.queue1.popleft())
        self.queue1, self.queue2 = self.queue2, self.queue1

    def pop(self) -> int:
        return self.queue1.popleft()

    def top(self) -> int:
        return self.queue1[0]

    def empty(self) -> bool:
        return not self.queue1
