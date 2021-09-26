"""
工具类的函数
"""
import numpy as np

def calculate_distance(matrix):
    """
    :param matrix: 输入的每行 1*2为UAV的位置
    :return: 输出为10*10连通性矩阵
    """
    lcn_matix = np.zeros((10,10))
    for i in range(9):
        j = i+1
        distance = ( (matrix[i][0]-matrix[j][0])**2 + (matrix[i][1]-matrix[j][1])**2 )**0.5
        if distance<250:
            lcn_matix[i][j] = 1
            lcn_matix[j][i] = 1
    return lcn_matix


def dfs(matrix):
    """
    判断LCN网络的联通性
    输入为10*10的LUAV的连通性矩阵
    """
    start = 1
    visited = set()
    visited.add(start)
    for i in range(10):
        for j in range(10):
            if matrix[i][j] == 1:
                if j not in visited:
                    visited.add(j)

    sum = len(visited)
    if sum == 10:
        return 0
    else:
        return 1
def judge_done(map_cover):
    counter_num = 0
    for i in range(40):
        for j in range(40):
            if(map_cover[i][j]==1):
                counter_num = counter_num+1
    if counter_num == 1600:
        return 1
    else:
        return 0
if "__name__" == "__main__":
    matrix = np.ones((10,10))
    print(1)
    print((dfs(matrix)))
