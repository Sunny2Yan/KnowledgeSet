"""树
二叉搜索树：每个树节点都是 左节点 < 根节点 < 右节点
平衡二叉树：二叉树每个节点的左右两个子树的高度差的绝对值不超过 1

叶子节点：没有子节点的节点

前序遍历：根节点——>左子树(根，左，右)——>右子树(根，左，右)  [1, 2, 4, 5, 3, 6, 7]
中序遍历：左子树(左，根，右)——>根节点——>右子树(左，根，右)  [4, 2, 5, 1, 6, 3, 7]
后续遍历：左子树(左，右，根)——>右子树(左，右，根)——>根节点  [4, 5, 2, 6, 7, 3, 1]
"""
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class TreeTraversal(object):
    @staticmethod
    def order_traversal(root: Optional[TreeNode]) -> list[int]:
        """二叉树的前（中、后）序遍历
        (leetcode 144 94 145) 给定一个二叉树的根节点root，返回它的前、中、后序遍历。
        思路：构建深度遍历函数，前序先输出根值，在分别遍历左右子树；中序先遍历左子树，再输出根值，最后遍历右子树；后续先分别遍历左右子树，在输出根值。
        时O(n); 空O(n)
        """
        res = []

        def dfs(node: Optional[TreeNode]) -> Optional[list[int]]:
            if not node:
                return
            res.append(node.val)  # preorder
            dfs(node.left)
            # res.append(root.val)  # inorder
            dfs(node.right)
            # res.append(root.val)  # postorder

        dfs(root)
        return res

    @staticmethod
    def level_order(root: Optional[TreeNode]) -> list[list[int]]:
        """二叉树的层序遍历
        (leetcode 102, 剑指offer 32) 逐层地，从左到右访问二叉树的所有节点。
        思路：队列实现bfs。即，定义一个队列来盛放树的结点，每次取出一个节点，获取节点值，并把该节点的左右子节点加入队列。
        时O(n); 空O(n)
        """
        if not root:
            return []
        res, queue = [], [root]
        # flag = 1  # (leetcode 103) 二叉树的锯齿形层序遍历

        while queue:
            row = []
            for _ in range(len(queue)):
                node = queue.pop(0)
                row.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(row)
            # res.append(row if flag % 2 == 1 else row[::-1])
            # flag += 1

        return res

    def invert_tree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """翻转二叉树
        (leetcode 226, 剑指offer 27) 给定二叉树的根节点root，翻转二叉树，并返回其根节点.
        思路：反转左右子节点，并递归他们（还可以采用层序遍历中的队列方法）
        时O(n); 空O(n)
        """
        if not root:
            return None

        root.left, root.right = root.right, root.left
        self.invert_tree(root.right)
        self.invert_tree(root.left)

        return root

    def max_depth(self, root: Optional[TreeNode]) -> int:
        """二叉树的最大深度
        (leetcode 104, 剑指offer 55) 给定二叉树root，返回其最大深度（层数）
        思路：递归法，max(递归左节点， 递归右节点)+1；或者采用上述层序遍历的方法。
        时O(n); 空O(n)
        """
        if not root:
            return 0

        return max(self.max_depth(root.left), self.max_depth(root.right)) + 1

    @staticmethod
    def min_depth(root: Optional[TreeNode]) -> int:
        """二叉树的最小深度
        (leetcode 111) 给定二叉树root，返回其最小深度（从根节点到最近叶子节点）
        思路：采用层序遍历的方法。
        时O(n); 空O(n)
        """
        if not root:
            return 0

        res, queue = 1, [root]
        while queue:
            for _ in range(len(queue)):
                node = queue.pop(0)
                if not node.left and not node.right:  # 重点
                    return res
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res += 1

        return res

    @staticmethod
    def binary_tree_paths(root: Optional[TreeNode]) -> list[str]:
        """二叉树的所有路径
        (leetcode 257) 按任意顺序 ，返回所有从根节点到叶子节点的路径
        思路：深度遍历搜索，遇到叶子节点则将路径加入到结果列表中，否则分别递归左右子节点
        时O(n); 空O(n)
        """
        res = []
        if not root:
            return res

        def dfs(node: Optional[TreeNode], temp: str):
            temp += f'{node.val}'
            if not node.left and not node.right:
                res.append(temp)
            if node.left:
                dfs(node.left, temp + '->')
            if node.right:
                dfs(node.right, temp + '->')

        dfs(root, '')
        return res

    @staticmethod
    def sum_numbers(root: Optional[TreeNode]) -> int:
        """求根节点到叶节点数字之和
        (leetcode 129) 给定二叉树的根节点root，从根节点到叶节点的每条路径都代表一个数字，求所有数字之和
        思路：同上深度遍历搜索，遇到叶子节点则将路径加入到结果列表中，否则分别递归左右子节点，最后对列表中的数求和
        时O(n); 空O(n)
        """
        if not root:
            return 0
        res = []

        def dfs(node: Optional[TreeNode], temp: int):
            temp = temp * 10 + node.val
            if not node.left and not node.right:
                res.append(temp)  # 局部变量不能使用+
            if node.left:
                dfs(node.left, temp)
            if node.right:
                dfs(node.right, temp)

        dfs(root, 0)
        return sum(res)

    @staticmethod
    def has_path_sum(root: Optional[TreeNode], target_sum: int) -> bool:
        """路径总和 I
        (leetcode 112) 判断树中是否存在根节点到叶子节点的路径和等于目标和target_sum
        思路：同上深度遍历搜索，如果节点是叶子节点且路径和等于target_sum，则将True加入res列表，最终返回res的最后一个元素。
        时O(n); 空O(n)
        """
        if not root:
            return False
        # 递归解法
        # if not root.left and not root.right and root.val == target_sum:
        #     return True
        # return (self.has_path_sum(root.left, target_sum - root.val)
        #         or self.has_path_sum(root.right, target_sum - root.val))
        res = [False]

        def dfs(node: Optional[TreeNode], temp: int):
            temp += node.val
            if not node.left and not node.right and temp == target_sum:
                res.append(True)
            if node.left:
                dfs(node.left, temp)
            if node.right:
                dfs(node.right, temp)

        dfs(root, 0)
        return res[-1]

    @staticmethod
    def path_sum(root: Optional[TreeNode], target_sum: int) -> list[list[int]]:
        """路径总和 II
        (leetcode 113) 找出所有从根节点到叶子节点路径总和等于给定目标和的路径
        思路：同上深度遍历搜索，但注意需要回溯。
        时O(n); 空O(n)
        """
        res = []
        if not root:
            return res

        def dfs(node: Optional[TreeNode], temp: list[int]):
            temp.append(node.val)
            if not node.left and not node.right and sum(temp) == target_sum:
                res.append(temp[:])  # 此处需要注意temp[:]
            if node.left:
                dfs(node.left, temp)
            if node.right:
                dfs(node.right, temp)
            temp.pop()  # 需要回溯

        dfs(root, [])
        return res


class TreeType(object):
    def __init__(self):
        pass

    def is_same_tree(self, p: Optional[TreeNode],
                     q: Optional[TreeNode]) -> bool:
        """相同的树
        (leetcode 100) 判断两棵树是否完全相同。
        思路：递归比较
        时O(n); 空O(1)
        """
        if not p and not q:
            return True
        elif not p or not q or p.val != q.val:
            return False
        else:
            return (self.is_same_tree(p.left, q.left) and
                    self.is_same_tree(p.right, q.right))

    @staticmethod
    def is_symmetric(root: Optional[TreeNode]) -> bool:
        """对称二叉树
        (leetcode 101, 剑指offer 28) 给定一个二叉树的根节点root，检查它是否轴对称
        思路：递归每一个左右节点
        时O(n); 空O(1)
        """
        if not root:
            return True

        def dfs(root_l: Optional[TreeNode], root_r: Optional[TreeNode]):
            if not root_l and not root_r:
                return True
            elif not root_l or not root_r or root_l.val != root_r.val:
                return False
            else:
                return (dfs(root_l.left, root_r.right) and
                        dfs(root_l.right, root_r.left))

        return dfs(root.left, root.right)

    @staticmethod
    def is_balanced(root: Optional[TreeNode]) -> bool:
        """平衡二叉树
        (leetcode 110, 剑指offer 55) 给定一个二叉树，判断它是否是高度平衡的二叉树
        思路：计算树的高度，递归左右子树，判断高度差小于1
        时O(n); 空O(n)
        """
        if not root:
            return True

        def tree_height(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            left_height = tree_height(node.left)
            right_height = tree_height(node.right)
            if left_height == -1 or right_height == -1 or abs(
                    left_height - right_height) > 1:  # -1 表示不是平衡二叉树
                return -1
            else:
                return max(left_height, right_height) + 1

        return tree_height(root) != -1
        # return abs(tree_height(root.left) - tree_height(root.right)) <= 1
        # and self.isBalanced(root.left) and self.isBalanced(root.right)
        # 时O(nlogn); 空O(n)

    def kth_smallest(self, root: Optional[TreeNode], k: int) -> int:
        """二叉搜索树中第K小的元素
        (leetcode 230; 剑指offer 54) 找出二叉搜索树中第k个最小元素（从1开始计数）。
        思路：二叉搜索树的的中序遍历为升序。
        时O(n); 空O(n)
        """
        self.res, self.k = 0, k

        def dfs(node: Optional[TreeNode]):
            if not node:
                return
            dfs(node.left)
            self.k -= 1
            if self.k == 0:
                self.res = node.val
            dfs(node.right)
        dfs(root)

        return self.res

    def lowest_common_ancestor_1(
            self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        """二叉搜索树的最近公共祖先
        (leetcode235, 剑指offer 68) 找出二叉搜索树中节点 p和q的最近公共节点
        思路：按照root<p遍历左子树，root>q遍历右子树，进行递归
        时O(n); 空O(1)
        """
        if root.val < p.val and root.val < q.val:
            return self.lowest_common_ancestor_1(root.right, p, q)
        elif root.val > p.val and root.val > q.val:
            return self.lowest_common_ancestor_1(root.left, p, q)
        else:
            return root

    def lowest_common_ancestor_2(
            self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        """二叉树的最近公共祖先
        (leetcode 236, 剑指offer 68) 找出二叉树中节点 p和q的最近公共节点
        思路：
        时O(n); 空O(n)
        """
        if not root or root == p or root == q:
            return root

        left = self.lowest_common_ancestor_2(root.left, p, q)
        right = self.lowest_common_ancestor_2(root.right, p, q)
        if left and right:
            return root

        return left if left else right

    def is_sub_tree(self, root: Optional[TreeNode],
                    subroot: Optional[TreeNode]) -> bool:
        """另一棵树的子树
        (leetcode 572, 剑指offer 26) 给定两棵二叉树，判断root中是否包含与subroot相同结构的子树
        思路：定义函数func判断两棵树是否相同，递归左右子树并使用func()判断是否相同
        时O(n^2); 空O(n)
        """
        if not root and not subroot:
            return True
        elif not root or not subroot:
            return False
        else:
            return (self.__same_tree(root, subroot) or
                    self.is_sub_tree(root.left, subroot) or
                    self.is_sub_tree(root.right, subroot))

    def __same_tree(self, node1, node2):
        if not node1 and not node2:
            return True
        elif not node1 or not node2:
            return False
        else:
            return (node1.val == node2.val and
                    self.__same_tree(node1.left, node2.left) and
                    self.__same_tree(node1.right, node2.right))


class TreeConstruction(object):
    def merge_trees(self, root1: Optional[TreeNode],
                    root2: Optional[TreeNode]) -> Optional[TreeNode]:
        """合并二叉树
        (leetcode 617) 将一棵覆盖到另一棵之上, 如果节点重叠，则加和为新节点的值；否则，不为null的节点为新节点值
        思路：深度遍历搜索（递归）
        时O(n); 空O(n)
        """
        if not root1 and not root2:
            return None
        elif not root1:
            return root2
        elif not root2:
            return root1
        else:
            root = TreeNode(root1.val + root2.val)
            root.left = self.merge_trees(root1.left, root2.left)
            root.right = self.merge_trees(root1.right, root2.right)

        return root

    def build_tree_1(self, preorder: list[int],
                     inorder: list[int]) -> Optional[TreeNode]:
        """从前序与中序遍历序列构造二叉树
        (leetcode 105, 剑指offer 07) 给定两个整数数组preorder和inorder，构造二叉树并返回根节点
        思路：分割preorder和inorder数组，进行原函数递归
        时O(n); 空O(n)
        """
        if not preorder:
            return None
        root = TreeNode(preorder[0])
        idx = inorder.index(preorder[0])
        root.left = self.build_tree_1(preorder[1:idx+1], inorder[:idx])
        root.right = self.build_tree_1(preorder[idx+1:], inorder[idx+1:])

        return root

    def build_tree_2(self, inorder: list[int],
                     postorder: list[int]) -> Optional[TreeNode]:
        """从中序与后序遍历序列构造二叉树
        (leetcode 106) 给定两个整数数组inorder和postorder，构造二叉树并返回根节点
        思路：分割inorder和postorder数组（注意分割点），进行原函数递归
        时O(n); 空O(n)
        """
        if not inorder:
            return None
        root = TreeNode(postorder[-1])
        idx = inorder.index(postorder[-1])
        root.left = self.build_tree_2(inorder[:idx], postorder[:idx])
        root.right = self.build_tree_2(inorder[idx+1:], postorder[idx:-1])
        return root


# 二叉搜索树的后序遍历 (剑指offer 33)
# 二叉搜索树与双向链表 (剑指offer 36)
