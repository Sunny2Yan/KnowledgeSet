"""排序算法

| 排序算法 | 平均时间复杂度 | 最坏复杂度 | 空间复杂度 | 稳定性 |
| :----: | :---------: | :------: | :------: | :---: |
| 冒泡排序 |   O(n^2)    |  O(n^2)  |  O(1)    |  稳定 |\
| 选择排序 |   O(n^2)    |  O(n^2)  |  O(1)    | 不稳定 |\
| 插入排序 |   O(n^2)    |  O(n^2)  |  O(1)    |  稳定 |\
| 快速排序 |   O(nlogn)  |  O(n^2)  | O(nlogn) | 不稳定 |\
| 归并排序 |   O(nlogn)  | O(nlogn) |  O(1)    |  稳定 |\
| 堆排序  |   O(nlogn)  |  O(nlogn) | O(1)     | 不稳定 |
"""


class Sorts(object):
    def __init__(self, nums: list[int]):
        self.nums = nums
        self.n = len(nums)

    def bubble(self):
        """冒泡排序
        比较相邻的元素。如果第一个比第二个大，就交换他们两个。(每轮得到一个最大的数)
        """
        if self.n <= 1:
            return self.nums
        for i in range(1, self.n):
            for j in range(self.n-i):
                if self.nums[j] > self.nums[j+1]:
                    self.nums[j], self.nums[j+1] = self.nums[j+1], self.nums[j]

        return self.nums

    def selection(self):
        """选择排序
        初始化一个最小（大）元素位置，遍历列表更新这个位置，如果有小于（大于）这个位置的值时，记录并交换。
        """
        if self.n <= 1:
            return self.nums
        for i in range(self.n-1):
            min_ = i
            for j in range(i+1, self.n):
                if self.nums[j] < self.nums[min_]:
                    min_ = j
            self.nums[i], self.nums[min_] = self.nums[min_], self.nums[i]

        return self.nums

    def insert(self):
        """插入排序
        类似于打扑克原理，定义一个关键点，左边是排序数组，关键点及右边是未排序数组，从未被排序的
        数组中抽取一个元素，插入到已被排序的数组中。
        """
        if self.n <= 1:
            return self.nums
        for i in range(1, self.n):
            key_point = self.nums[i]
            j = i - 1
            while j >= 0 and key_point < self.nums[j]:
                self.nums[j+1] = self.nums[j]
                j -= 1
            self.nums[j+1] = key_point  # 本是第j个位置，但上面-=1了

        return self.nums

    def quick(self, nums):
        """快速排序
        选择第一个元素作为基准值（pivot），并将列表分成两个子列表：一个包含小于等于基准值的元素，
        另一个包含大于基准值的元素。然后，我们递归地对这两个子列表进行快速排序
        """
        if len(nums) <= 1:
            return nums
        else:
            pivot = nums[0]
            left = [num for num in nums[1:] if num <= pivot]
            right = [num for num in nums[1:] if num > pivot]
            return self.quick(left) + [pivot] + self.quick(right)

    def merge(self, nums):
        """归并排序
        采用二分法的思想，递归的对数组进行分区（左、右两个数组），每次对左右两个数组进行排序，
        并返回排序后的数组。
        """
        if len(nums) <= 1:
            return nums
        mid = len(nums) // 2
        left_nums = self.merge(nums[:mid])
        right_nums = self.merge(nums[mid:])
        return self.__merge_sub(left_nums, right_nums)

    @staticmethod
    def __merge_sub(left_nums, right_nums):
        left, right, res = 0, 0, []
        while left < len(left_nums) and right < len(right_nums):
            if left_nums[left] <= right_nums[right]:
                res.append(left_nums[left])
                left += 1
            else:
                res.append(right_nums[right])
                right += 1
        res += left_nums[left:]
        res += right_nums[right:]

        return res

    def heap(self):
        """堆排序
        大顶堆：每个结点的值都大于或等于其左右孩子结点的值（用于升序）；
        小顶堆：每个结点的值都小于或等于其左右孩子结点的值（用于降序）；
        """
        if self.n <= 1:
            return self.nums
        # 子节点找父节点：n->(n-1)//2，子节点位置为L-1
        for i in range((self.n - 2) // 2, -1, -1):
            # 1.建立初始堆,由于end不好找，让他一直为列表末尾
            self.__heap_adjust(self.nums, i, self.n-1)
        # 进行n-1趟排序,j指向当前最后一个位置
        for j in range(self.n-1, -1, -1):
            # 根与最后一个元素交换
            self.nums[0], self.nums[j] = self.nums[j], self.nums[0]
            # 2.重新建堆，j-1是新的end
            self.__heap_adjust(self.nums, 0, j-1)

        return self.nums

    @staticmethod
    def __heap_adjust(nums, start, end):
        """先把数组构造成大顶堆（父节点大于子节点），然后把堆顶和数组最后一个元素交换；
        再对前 n-1个元素进行堆排序。此处调整为大顶堆
        """
        temp = nums[start]  # 根节点
        i = start
        j = 2 * i + 1  # 根的左子树，右子树2*i+2
        while j <= end:
            if j + 1 <= end and nums[j] < nums[j + 1]:  # 若左子树小于右子树，j指向右子树
                j += 1
            if temp < nums[j]:  # 如果子节点大于根节点，则交换。
                nums[i] = nums[j]
                i = j  # 往下看一层
                j = 2 * i + 1
            else:
                break
        nums[i] = temp  # temp大，把temp放到i的位置即可/i为最后一个节点，没有j，就需要把temp放到i位置


    # 7.希尔排序
    # 8.计数排序
    # 9.桶排序
    # 10.基数排序


if __name__ == '__main__':
    nums_1 = [2]
    nums_2 = [5, 7, 2, 4, 3, 1, 6, 9, 8]

    sort = Sorts(nums_2)
    print("bubble:", sort.bubble())
    print("selection:", sort.selection())
    print("insert:", sort.insert())
    print("quick:", sort.quick(nums_2))
    print("merge:", sort.merge(nums_2))
    print("heap:", sort.heap())