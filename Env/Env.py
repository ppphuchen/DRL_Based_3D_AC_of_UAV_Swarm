import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from collections import namedtuple
import random
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.ticker as mtick

from collections import namedtuple
from itertools import count
from PIL import Image
from Arguments.args import arglists
import  numpy as np
class Env(object):
    """
    环境
    """
    def __init__(self, arguments):
        """
        创造一个环境
        :param L_UAV_num: (int) 环境中LUAV的数量
        :param L_power: (float) 环境中LUAV的发送功率
        :param F_power: (float) 环境中FUAV的发送功率
        :param can_comm: (float) 环境中LUAV之间能够通信的最小功率
        :param patch_num: (int) 环境中子群的数量
        :param total_UAV_num: (int) 环境中UAV的总数量
        :param re_cover_rate: (float) LUAV的重复覆盖率
        :param patch_fuav_num: (in) 子群中FUAV的数量
        :param L_comm_radius: (int) LUAV的通信范围
        :param F_comm_radius: (int) FUAV的通信范围
        :param gain_r: (float) LUAV的接收增益
        :param gain_t: (float) LUAV的发送增益
        :param average_collect_time: (int) 子群飞行每米后所需要的停下来收集数据的时间
        :param max_sub_group_cover_time: (int) 所有的子群总覆盖时间中的最大值,通过遍历比较得出，这里只给一个初始值
        :param H:(int) UAV的默认飞行高度
        :param L_x: (int) 3D模型的横轴大小
        :param L_y: (int) 3D模型的纵轴大小
        :param S_x: (int) 采样方格的横轴大小
        :param S_y: (int) 采样方格的纵轴大小
        """
        self.L_UAV_num = arguments.L_UAV_num
        self.L_power = arguments.L_power
        self.F_power = arguments.F_power
        self.can_comm = arguments.can_comm
        self.patch_num = arguments.patch_num
        self.F_line_v = arguments.F_line_v
        self.L_comm_radius = arguments.L_comm_radius
        self.F_comm_radius = arguments.F_comm_radius
        self.re_cover_rate = arguments.re_cover_rate
        self.H = arguments.H
        self.L_x = arguments.L_x
        self.L_y = arguments.L_y
        self.S_x = arguments.S_x
        self.S_y = arguments.S_y
        self.max_sub_group_cover_time = arguments.max_sub_group_cover_time
        self.L_UAV_pos = np.zeros((10, 10))
        #LUAV的位置矩阵
        self.gain_r = arguments.gain_r
        #UAV的接收增益
        self.gain_t = arguments.gain_t
        #UAV的发送增益
        self.F_UAV_pos = np.zeros((100, 3))
        #第i个LUAV所在的子群中第j个FUAV的位置
        self.neighbors = []
        #第i个LUAV的邻居数量
        self.L_connection = np.zeros((100, 100))
        #LUAV的连通性矩阵
        self.d = self.L_x/(self.L_x / self.S_x)
        #子群的起始位置的初始距离
        self.patch_fly_d = np.zeros((100, 3, 100))
        #第i个子群里的第j个FUAV在第q个子群里的飞行距离
        self.patch_collect_time = np.zeros((100,100))
        #第i个子群在第q个patch的总收集时间
        self.sub_sub_group_fly_time = np.zeros((100,100))
        #第i个子群从第j个小块到第j+1个小块的飞行时间
        self.total_sub_group_fly_time = []
        #第i个子群总的飞行时间
        self.total_sub_group_cover_time = []
        #第i个子群的总的覆盖时间
        self.max_sub_group_cover_time = arguments.max_sub_group_cover_time
        #所有的子群总覆盖时间中的最大值
        self.patch_covered_flag = np.zeros((100, 100))
        #patch(i,j) 为0代表patch(i,j)没有被覆盖，为1代表patch(i,j)已经被覆盖

        self.init_pos = np.zeros((10,2))
        # 第i行0列代表
        for i in range(10):
            self.init_pos[i][0] = (i) * 100
            self.init_pos[i][1] = 0
            self.L_UAV_pos[i][0] = self.init_pos[i][0]
            self.L_UAV_pos[i][1] = self.init_pos[i][1]
        # 子群的起始位置, (0,0) (100,0) (200,0) ...(900,0)

        self.map_cover = np.zeros((40, 40))
        for i in range(10):
            self.map_cover[0][i*4] = 1
        #地图的覆盖情况

        self.luav_history_pos = []
        for i in range(10):
            self.luav_history_pos.append(collections.deque(maxlen=3))
            self.luav_history_pos[i].append(torch.tensor([ self.init_pos[i][0], self.init_pos[i][1] ]))
        #维护每个luav的过去两个step的历史位置和当前位置的队列

        self.luav_point_pos = np.zeros((1000,1000))
        for i in range(10):
            self.luav_point_pos[i][0] = i*4    #luav的格子横坐标
            self.luav_point_pos[i][1] = 0      #luav的格子纵坐标

        self.render_pos = np.zeros((10000, 10000)) #用于作图的坐标，每两列为LUAV已经覆盖的位置
        for i in range(10):
            self.render_pos[0][i*2] = self.L_UAV_pos[i][0]
            self.render_pos[0][i*2+1] = self.L_UAV_pos[i][1]

        self.render_row = np.ones((10, 1))  # 维护一个10行1列关于LUAV需要绘制的坐标的行数的数组，便于作图
    def reset(self, arguments):

        self.init_pos = np.zeros((10, 2))
        for i in range(10):
            self.init_pos[i][0] = (i) * 100
            self.init_pos[i][1] = 0
            self.L_UAV_pos[i][0] = self.init_pos[i][0]
            self.L_UAV_pos[i][1] = self.init_pos[i][1]
        # 子群的起始位置, (0,0) (100,0) (200,0) ...(900,0)

        self.map_cover = np.zeros((40, 40))
        for i in range(10):
            self.map_cover[0][i * 4] = 1

        self.luav_history_pos = []
        for i in range(10):
            self.luav_history_pos.append(collections.deque(maxlen=3))
            self.luav_history_pos[i].append(torch.tensor([self.init_pos[i][0], self.init_pos[i][1]]))

        # 维护每个luav的过去两个step的历史位置和当前位置的队列

        self.luav_point_pos = np.zeros((1000, 1000))
        for i in range(10):
            self.luav_point_pos[i][0] = i * 4  # luav的格子横坐标
            self.luav_point_pos[i][1] = 0  # luav的格子纵坐标

        self.render_pos = np.zeros((10000, 10000))  # 用于作图的坐标，每两列为LUAV已经覆盖的位置
        for i in range(10):
            self.render_pos[0][i * 2] = self.L_UAV_pos[i][0]
            self.render_pos[0][i * 2 + 1] = self.L_UAV_pos[i][1]

        self.render_row = np.ones((10, 1)) # 维护一个10行1列关于LUAV需要绘制的坐标的行数的数组，便于作图

    def sys_model_reset(self, arguments):
        """系统模型参数的重置,L_UAV的数量,L_UAV的通信功率，F_UAV的通信功率，L_UAV之间能够进行通信的阈值功率"""

        self.L_UAV_num = arguments.L_UAV_num
        self.L_power = arguments.L_power
        self.F_power = arguments.F_power
        self.can_comm = arguments.can_comm
        self.patch_num = arguments.patch_num

    def comm_connection_model_rest(self, arguments):
        """通信连接模型的重置,LUAV的发送增益，传输增益，FUAV的位置"""

        self.gain_r = arguments.gain_r
        self.gain_t = arguments.gain_t
        self.F_UAV_pos = np.zeros((100, 3))
    def LCN_model_reset(self, arguments):
        """LCN网络参数的重置，"""

        self.neighbors = []
        self.L_connection = np.zeros((100, 100))
        self.init_pos = []
    def time_consume_and_comm_cover(self, arguments):
        """时间消耗和通信覆盖参数的重置"""

        self.init_pos = []
        self.d = self.L_x / (self.L_x / self.S_x)
        self.patch_fly_d = np.zeros((100, 3, 100))
        self.patch_collect_time = np.zeros((100, 100))
        self.sub_sub_group_fly_time = np.zeros((100, 100))
        self.total_sub_group_fly_time = []
        self.total_sub_group_cover_time = []
        self.max_sub_group_cover_time = arguments.max_sub_group_cover_time
        self.patch_covered_flag = np.zeros((100, 100))

    def render(self):
        """
        由self.render_pos
        做出轨迹图
        :return:
        """
        x0 = self.render_pos[0:int(self.render_row[0][0]), 0]
        y0 = self.render_pos[0:int(self.render_row[0][0]), 1]
        x1 = self.render_pos[0:int(self.render_row[1][0]), 2]
        y1 = self.render_pos[0:int(self.render_row[1][0]), 3]
        x2 = self.render_pos[0:int(self.render_row[2][0]), 4]
        y2 = self.render_pos[0:int(self.render_row[2][0]), 5]
        x3 = self.render_pos[0:int(self.render_row[3][0]), 6]
        y3 = self.render_pos[0:int(self.render_row[3][0]), 7]
        x4 = self.render_pos[0:int(self.render_row[4][0]), 8]
        y4 = self.render_pos[0:int(self.render_row[4][0]), 9]
        x5 = self.render_pos[0:int(self.render_row[5][0]), 10]
        y5 = self.render_pos[0:int(self.render_row[5][0]), 11]
        x6 = self.render_pos[0:int(self.render_row[6][0]), 12]
        y6 = self.render_pos[0:int(self.render_row[6][0]), 13]
        x7 = self.render_pos[0:int(self.render_row[7][0]), 14]
        y7 = self.render_pos[0:int(self.render_row[7][0]), 15]
        x8 = self.render_pos[0:int(self.render_row[8][0]), 16]
        y8 = self.render_pos[0:int(self.render_row[8][0]), 17]
        x9 = self.render_pos[0:int(self.render_row[9][0]), 18]
        y9 = self.render_pos[0:int(self.render_row[9][0]), 19]

        myparams = {
            'axes.labelsize': '20',
            'xtick.labelsize': '18',
            'ytick.labelsize': '18',
            'lines.linewidth': 1.3,
            'legend.fontsize': '18',
            'font.family': 'Times New Roman',
            'figure.figsize': '7, 7',  # 图片尺寸
            'grid.alpha': 0.1

        }

        plt.style.use("seaborn-deep")
        pylab.rcParams.update(myparams)
        '''
        params = {
        'axes.labelsize': '35',
        'xtick.labelsize': '27',
        'ytick.labelsize': '27',
        'lines.linewidth': 2,
        'legend.fontsize': '27',
        'figure.figsize': '12, 9'  # set figure size
        }

        pylab.rcParams.update(params)  # set figure parameter
        # line_styles=['ro-','b^-','gs-','ro--','b^--','gs--']  #set line style
        '''
        plt.plot(x0, y0, marker='^', markersize=5)
        plt.plot(x1, y1, marker='^', markersize=5)
        plt.plot(x2, y2, marker='^', markersize=5)
        plt.plot(x3, y3, marker='^', markersize=5)
        plt.plot(x4, y4, marker='^', markersize=5)
        plt.plot(x5, y5, marker='^', markersize=5)
        plt.plot(x6, y6, marker='^', color = 'blue', markersize=5)
        plt.plot(x7, y7, marker='^', color ='green', markersize=5)
        plt.plot(x8, y8, marker='^', color = 'red',markersize=5)
        plt.plot(x9, y9, marker='^', color = 'magenta',markersize=5)

        ax_values = [0, 1000, 0, 1000]
        plt.axis(ax_values)
        plt.axhline()
        plt.axvline()

        plt.legend(loc="lower right")  # set legend location
        plt.ylabel('y_coordinate')  # set ystick label
        plt.xlabel('x_coordinate')  # set xstck label
        self.rac_num = float(str(self.map_cover).count("1"))
        self.rac = float(str(self.map_cover).count("1")) / 1600
        LUAV_num = 0
        for i in range(10):
            LUAV_num = LUAV_num + int(self.render_row[i])

        self.rap = float(LUAV_num) / 1600 - self.rac
        print("覆盖率=%d  重复覆盖率=%d ", self.rac_num, LUAV_num)
        plt.show()


if __name__ == "__main__":
    args = arglists()
    new_Env = Env(args)
    new_Env.render()

