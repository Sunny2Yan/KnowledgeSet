"""图

"""


class GraphDFS(object):
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


class Graph(object):
    def __init__(self):
        ...

