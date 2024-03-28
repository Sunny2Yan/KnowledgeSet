"""图
图存储：邻接矩阵(adjacency matrix)、邻接表(adjacency list)、邻接多重表(adjacency multilists)
"""
import collections


class GraphDFS(object):
    """深度优先搜索(DFS，depth first search)是一个递归过程，有回退过程。步骤如下：
    1. 构建dfs函数：dfs(si, sj, cnt )。(si, sj)表示目前所在位置，cnt表示产生的花费；
    2. 确定终止条件：si=di，sj=dj，cnt=t。(di,dj)表示结束位置，t表示产生的花费；
    3. 中间搜索按照上、右、下、左顺时针的顺序进行搜索；
    4. 每次搜索后进行回溯，回复原图元素。
    """
    def __init__(self):
        ...

    @staticmethod
    def num_islands(grid: list[list[str]]) -> int:
        """岛屿数量（包含边界）
        (leetcode 200) 给定一个'1'为陆地，'0'为水的二维网格，计算网格中岛屿的数量。
        思路：如果一个位置为'1'，以其为起始节点开始深度优先搜索，分别向左、上、右、下四个方向进行扩展。在搜索的过程中，搜索过的替换为'0'
        时O(n^2); 空(1)
        """
        def dfs(x: int, y: int) -> bool:
            if x < 0 or y < 0 or x >= len(grid) or y >= len(grid[0]):
                return True
            if grid[x][y] != '1':
                return True
            grid[x][y] = '0'

            return (dfs(x - 1, y) and dfs(x + 1, y) and
                    dfs(x, y - 1) and dfs(x, y + 1))  # 每一个方向都不是土地

        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1' and dfs(i, j):
                    res += 1

        return res

    @staticmethod
    def closed_island(grid: list[list[int]]) -> int:
        """统计封闭岛屿的数目（不包含边界）
        (leetcode 1254) 二维矩阵由0（土地）和1（水）组成。封闭岛是一个完全由1包围（左、上、右、下）的岛，返回封闭岛屿的数目
        思路：如果一个位置为0，则以其为起始节点开始深度优先搜索，分别向左、上、右、下四个方向进行扩展。在搜索的过程中，搜索过的0都会被替换为-1
        时O(n^2); 空O(1)
        """
        def dfs(x: int, y: int) -> bool:
            if x < 0 or y < 0 or x >= len(grid) or y >= len(grid[0]):
                return False
            if grid[x][y] != 0:
                return True

            grid[x][y] = -1  # 访问过的记为 -1
            left, right = dfs(x - 1, y), dfs(x + 1, y)
            bottom, top = dfs(x, y - 1), dfs(x, y + 1)
            return left and right and bottom and top  # 每一个方向都不是土地

        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 0 and dfs(i, j):
                    res += 1

        return res

    @staticmethod
    def island_perimeter(grid: list[list[int]]) -> int:
        """岛屿周长
        (leetcode 463) 给定二维网格grid，其中grid[i][j]=1表示陆地，grid[i][j]=0表示水域，计算陆地的周长
        思路：如果某个位置为1，从这个位置进行深度优先搜索，分别考虑边界、靠水、靠访问过的一侧
        时O(n^2); 空O(n^2)
        """
        def dfs(x: int, y: int) -> int:
            if x < 0 or y < 0 or x >= len(grid) or y >= len(grid[0]):  # 边界
                return 1
            if grid[x][y] == 0:  # 边缘靠水
                return 1
            if grid[x][y] == -1:  # 边缘靠访问过的陆地
                return 0
            grid[x][y] = -1  # 记录访问过的陆地
            return dfs(x - 1, y) + dfs(x + 1, y) + dfs(x, y - 1) + dfs(x, y + 1)

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    return dfs(i, j)  # 只有一个岛屿

        return 0

    @staticmethod
    def max_area_of_island(grid: list[list[int]]) -> int:
        """岛屿的最大面积
        (leetcode 695) 给定一个二进制矩阵grid，1代表土地，0代表水，岛屿的面积为1的数目。返回最大的岛屿面积。
        思路：同上
        时O(n^2); 空O(n^2)
        """
        def dfs(x: int, y: int) -> int:
            if x < 0 or y < 0 or x >= len(grid) or y >= len(grid[0]):
                return 0
            if grid[x][y] != 1:
                return 0
            grid[x][y] = 0  # 访问过的记为0
            area = 1
            area += dfs(x - 1, y) + dfs(x + 1, y) + dfs(x, y - 1) + dfs(x, y + 1)
            return area

        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    res = max(res, dfs(i, j))

        return res


class GraphBFS:
    """广度优先搜索(BFS，Breadth First Search)是一个分层的搜索过程，没有回退过程，是非递归的。
    1. 定义状态数组 visited=[(i, j)]，用来存储各顶点的访问状态，避免重复；
    2. 从头取出队列中顶点，依次访问该顶点的邻接顶点，并入队列，直至队列为空；
    3. 遍历图的每一个元素，对每一个元素进行广度优先搜索
    """
    def __init__(self):
        ...

    @staticmethod
    def num_islands(grid: list[list[str]]) -> int:
        if len(grid) == 0:
            return 0

        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1":
                    res += 1
                    grid[i][j] = "0"  # 访问过的
                    visited = collections.deque([(i, j)])
                    while visited:
                        row, col = visited.popleft()
                        for x, y in [(row - 1, col), (row + 1, col),
                                     (row, col - 1), (row, col + 1)]:
                            if (0 <= x < len(grid) and 0 <= y < len(grid[0])
                                    and grid[x][y] == "1"):  # 且该点未被访问
                                visited.append((x, y))
                                grid[x][y] = "0"

        return res


class Graph(object):
    def __init__(self):
        ...

