from typing import Optional


class ListNode:
    def __init__(self, x, _next=None):
        self.val = x
        self.next = _next


# 1.排序链表
class SortLink(object):
    def sort_list(self, head: ListNode) -> ListNode:
        """排序链表
        (leetcode 148) 按升序排列链表head并返回排序后的链表
        思路：归并法排序，采用快慢指针来找链表中点位置，slow.next=None为链表前部分，slow.next为链表后部分
        时O(nlogn); 空O(logn)
        """
        if not head or not head.next:
            return head

        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        slow.next, mid = None, slow.next  # 分别为head的前一半与后一半

        return self.__merge(self.sort_list(head), self.sort_list(mid))

    @staticmethod
    def __merge(left_list, right_list):
        cur = new_list = ListNode(0)

        while left_list and right_list:
            if left_list.val <= right_list.val:
                cur.next = left_list
                left_list = left_list.next
            else:
                cur.next = right_list
                right_list = right_list.next
            cur = cur.next
        cur.next = left_list if left_list else right_list

        return new_list.next

    @staticmethod
    def odd_even_list(head: ListNode) -> ListNode:
        """奇偶链表重排
        (leetcode 328) 将链表的奇数位节点和偶数位节点分别放在一起，如[1,2,3,4,5]->[1,3,5,2,4]
        思路：新建奇偶指针和存放偶数位置的链表，原链表存放奇数位置，最后再组合
        时O(n); 空O(1)
        """
        if not head or not head.next or not head.next.next:
            return head

        odd = head  # 奇数位置指针
        even = even_list = head.next  # 偶数位置指针和存放偶数位置的新链表
        while even and even.next:
            odd.next = even.next
            odd = odd.next

            even.next = odd.next
            even = even.next
        odd.next = even_list  # 两链表拼接

        return head

    def reverse_list(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """反转链表（全部反转）
        (leetcode 206) 反转链表，并返回反转后的链表.
        时O(n); 空O(1)
        """
        new_node = None
        while head:
            temp = head.next
            head.next = new_node  # 将head当前节点指向new_node，此时的head为新链表

            new_node = head  # 更新新链表
            head = temp  # 重新赋值head，继续访问

        return new_node

    @staticmethod
    def reverse_between(head: Optional[ListNode], left: int,
                        right: int) -> Optional[ListNode]:
        """反转链表（区间反转）
        (leetcode 92) 反转从位置left到位置right的链表节点。
        时O(n); 空O(1)
        """
        new_head = ListNode(None)
        new_head.next = head

        pre = new_head  # 始终指向需要反转的前一个位置
        for _ in range(1, left):
            pre = pre.next

        cur = pre.next  # 指向需要反转的位置
        for _ in range(left, right):
            temp = cur.next

            cur.next = temp.next
            temp.next = pre.next
            pre.next = temp

        cur = cur.next
        return new_head.next

    # def reverse_group(self, head: ListNode, k: int) -> ListNode:
    #     """反转链表（分组反转）
    #     (leetcode 25)
    #     时O(n); 空O(1)
    #     """
    #     new_head = ListNode(None)
    #     new_head.next = head
    #     prev = new_head
    #     while new_head:
    #         # 1.查看剩余部分长度是否大于等于k，大于k的将tail移到组末尾
    #         tail = prev
    #         for _ in range(k):
    #             tail = tail.next
    #             if not tail:
    #                 return new_head.next
    #         next_prev = tail.next
    #         # 2.反转子链表
    #         head, tail = self.__reverse(head, tail)
    #         # 3.把子链表装回原链表
    #         prev.next = head
    #         tail.next = next_prev
    #         prev = tail
    #         head = tail.next
    #     return new_head.next
    #
    # @staticmethod
    # def __reverse(head: ListNode, tail: ListNode):
    #     """反转子链表并返回新的头和尾"""
    #     prev = tail.next  # 此时tail就是新的链表，是前面反转后的链表
    #     cur = head
    #     while prev != tail:
    #         temp = cur.next
    #         cur.next = prev
    #         prev = cur
    #         cur = temp
    #     return tail, head

    def is_palindrome(self, head: Optional[ListNode]) -> bool:
        """回文链表
        (leetcode 234) 判断链表是否为回文链表，如果是返回true，否则返回false。
        思路：将链表的后一半反转，并同前一半逐个节点比较值是否相等。
        时O(n); 空O(1)
        """
        slow, fast = head, head
        while fast and fast.next:  # 为什么不全部反转再比较？全部反转会改动原head，无法比较
            slow = slow.next
            fast = fast.next.next

        slow = self.reverse_list(slow)

        while slow:
            if head.val != slow.val:
                return False
            else:
                head = head.next
                slow = slow.next

        return True


# 2.合并链表
class MergeLink(object):
    @staticmethod
    def merge_2_lists(head1: ListNode, head2: ListNode) -> ListNode:
        """合并两个排序链表
        (leetcode 21) 将两个升序链表合并为一个新的升序链表并返回.
        时O(n+m); 空O(1)
        """
        if not head1:
            return head2
        if not head2:
            return head1

        new_node = cur = ListNode(0)
        while head1 and head2:
            if head1.val <= head2.val:
                cur.next = head1
                head1 = head1.next
            else:
                cur.next = head2
                head2 = head2.next
            cur = cur.next
        cur.next = head1 if head1 else head2

        return new_node.next

    def merge_k_lists(self, lists: list[ListNode]) -> ListNode:
        """合并K个升序链表
        (leetcode 23) 将所有链表合并到一个升序链表中，返回合并后的链表
        时O(nlogn); 空O(logn)
        """
        return self.__divide_merge(lists, 0, len(lists)-1)

    def __divide_merge(self, lists: list[ListNode], left: int, right: int):
        """二分法两两合并"""
        if left > right:  # <正常递归，=结束并返回，>忽略并返回None
            return None
        elif left == right:
            return lists[left]

        mid = left + (right - left) // 2
        return self.merge_2_lists(
            self.__divide_merge(lists, left, mid),
            self.__divide_merge(lists, mid + 1, right))


# 3.删除链表
class DeleteLink(object):
    @staticmethod
    def delete_node(head: ListNode, val: int) -> ListNode:
        """移除链表元素
        (leetcode 203) 删除链表中所有满足Node.val==val的节点.
        时O(n); 空O(1)
        """
        new_head = ListNode(None)
        new_head.next = head
        cur = new_head
        while cur.next:
            if cur.next.val == val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return new_head.next

    @staticmethod
    def remove_nth_from_end(head: ListNode, n: int) -> Optional[ListNode]:
        """删除链表的倒数第n个节点
        (leetcode 19) 删除链表的倒数第 n 个结点
        思路：双指针，快指针遍历完后，慢指针的下一个是倒数第n个节点。
        时O(n); 空O(1)
        """
        new_head = ListNode(0, head)  # 确保slow的下一个数是倒数第n个
        slow, fast = new_head, head
        for _ in range(n):
            fast = fast.next

        while fast:
            slow = slow.next
            fast = fast.next

        slow.next = slow.next.next

        return new_head.next

    @staticmethod
    def delete_duplicates_1(head: ListNode) -> Optional[ListNode]:
        """删除有序链表中的重复元素(保留一个)
        (leetcode 83) 删除所有重复的元素，使每个元素只出现一次
        时O(n); 空O(1)
        """
        if not head or not head.next:
            return head
        cur = head
        while cur.next:
            if cur.val == cur.next.val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return head

    @staticmethod
    def delete_duplicates_2(head: ListNode) -> Optional[ListNode]:
        """删除有序链表中的重复元素(不保留)
        (leetcode 82) 已排序的链表的头head， 删除其所有重复数字的节点，只留下不同的数字。
        时O(n); 空O(1)
        """
        if not head:
            return None
        new_head = ListNode(None, head)
        cur = new_head

        while cur.next and cur.next.next:
            if cur.next.val == cur.next.next.val:
                temp = cur.next.val
                # 将所有相同的都跳过
                while cur.next != None and cur.next.val == temp:
                    cur.next = cur.next.next
            else:
                cur = cur.next
        return new_head.next


# 4.环形链表
class RingList(object):
    @staticmethod
    def find_first_common_node(head1: ListNode,
                               head2: ListNode) -> Optional[ListNode]:
        """两链表的第一个公共节点
        (leetcode 160) 返回两个单链表相交的起始节点，如果不存在相交节点，返回null
        思路：拼接两单链表，相同位置值相同的节点为公共节点。[12678,34678]->[1267834678, 3467812678]
        时O(n+m); 空O(1)
        """
        if not head1 or not head2:
            return None

        cur1, cur2 = head1, head2
        # TODO 为什么 cur1.val != cur2.val 不对？
        while cur1 != cur2:
            cur1 = cur1.next if cur1 else head2
            cur2 = cur2.next if cur2 else head1
        return cur1

    @staticmethod
    def has_cycle(head: ListNode) -> bool:
        """环形链表 I
        (leetcode 141) 判断链表中是否有环，如果存在环，返回true，否则返回false。
        思路：采用快慢指针，快指针走两步，慢指针走一步，如果存在环形，则快慢指针必相遇，否则不相遇。
        时O(n); 空O(1)
        """
        fast, slow = head, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                return True
        return False

    def entry_node_of_loop(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """环形链表 II
        (leetcode 142) 返回链表开始入环的第一个节点。如果链表无环，则返回null.
        思路：设fast在环中走了n圈，slow在环中走了m圈相遇。进入环之前的距离为x，环入口到相遇点的
        距离为y，相遇点到环入口的距离为z。则快指针一共走了x+n(y+z)+y步，慢指针一共走了
        x+m(y+z)+y，此时快指针走的倍数是慢指针的两倍，则x+n(y+z)+y=2(x+m(y+z)+y)，这时
        x+y=(n-2m)(y+z)，因为环的大小是y+z，说明从链表头经过环入口到达相遇地方经过的距离等于
        环的大小整数倍。x=(2m-n-1)(y+z)+z，从而从起点和相遇点出发的指针必相遇再入口节点。
        时O(n); 空O(1)
        """
        meet_head = self.__has_loop(head)
        if not meet_head:
            return None

        new_head = head
        while new_head != meet_head:
            new_head = new_head.next
            meet_head = meet_head.next
        return meet_head

    @staticmethod
    def __has_loop(head: Optional[ListNode]) -> Optional[ListNode]:
        """判断是否有环并返回相遇节点，无环返回None。"""
        if not head:
            return None
        fast, slow = head, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:  # 有环则相遇
                return slow
        return None  # 无环输出None


# 5.链表运算
class OperationList(object):
    @staticmethod
    def add_two_numbers_1(l1: Optional[ListNode],
                          l2: Optional[ListNode]) -> Optional[ListNode]:
        """两数相加 I（链表的倒序相加）
        (leetcode 2) 两个非空链表表示两个非负的整数，其每位数字都按照逆序存储。请你将两个数相加，并返回一个新的链表。
        思路：创建新链表存储结果，定义flag存储进位数，然后s=l1.val+l2.val+flag
        时O(n); 空O(1)
        """
        cur = new_head = ListNode(0)
        if not l1 and not l2:
            return new_head

        flag = 0
        while l1 or l2 or flag != 0:  # flag != 0确保最后一位满10能够进1
            s = (l1.val if l1 else 0) + (l2.val if l2 else 0) + flag
            flag = s // 10
            cur.next = ListNode(s % 10)
            cur = cur.next

            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None

        return new_head.next

    def add_two_numbers_2(self, l1: Optional[ListNode],
                          l2: Optional[ListNode]) -> Optional[ListNode]:
        """两数相加 II（正序相加）
        (leetcode 445) 向左进位：937+63=1000
        思路：将链表逆序在相加，再对新链表逆序即可。
        时O(n); 空O(1)
        """
        l1 = self.__reverse(l1)
        l2 = self.__reverse(l2)
        new_head = self.add_two_numbers_1(l1, l2)

        return self.__reverse(new_head)  # 最后将结果反转

    @staticmethod
    def __reverse(head):
        new_head = None
        while head:
            temp = head.next
            head.next = new_head

            new_head = head
            head = temp

        return new_head