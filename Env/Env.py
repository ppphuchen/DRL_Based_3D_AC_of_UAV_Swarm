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
        self.L_UAV_pos = []
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
        self.init_pos = []
        #子群的起始位置
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


    def reset(self, arguments):
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
        self.L_UAV_pos = []
        # LUAV的位置矩阵
        self.gain_r = arguments.gain_r
        # UAV的接收增益
        self.gain_t = arguments.gain_t
        # UAV的发送增益
        self.F_UAV_pos = np.zeros((100, 3))
        # 第i个LUAV所在的子群中第j个FUAV的位置
        self.neighbors = []
        # 第i个LUAV的邻居数量
        self.L_connection = np.zeros((100, 100))
        # LUAV的连通性矩阵
        self.init_pos = []
        # 子群的起始位置
        self.d = self.L_x / (self.L_x / self.S_x)
        # 子群的起始位置的初始距离
        self.patch_fly_d = np.zeros((100, 3, 100))
        # 第i个子群里的第j个FUAV在第q个子群里的飞行距离
        self.patch_collect_time = np.zeros((100, 100))
        # 第i个子群在第q个patch的总收集时间
        self.sub_sub_group_fly_time = np.zeros((100, 100))
        # 第i个子群从第j个小块到第j+1个小块的飞行时间
        self.total_sub_group_fly_time = []
        # 第i个子群总的飞行时间
        self.total_sub_group_cover_time = []
        # 第i个子群的总的覆盖时间
        self.max_sub_group_cover_time = arguments.max_sub_group_cover_time
        # 所有的子群总覆盖时间中的最大值
        self.patch_covered_flag = np.zeros((100, 100))
        # patch(i,j) 为0代表patch(i,j)没有被覆盖，为1代表patch(i,j)已经被覆盖


    def sys_model_reset(self, arguments):
        """系统模型参数的重置,L_UAV的数量,L_UAV的通信功率，F_UAV的通信功率，L_UAV之间能够进行通信的阈值功率"""

        self.L_UAV_num = arguments.L_UAV_num
        self.L_power = arguments.L_power
        self.F_power = arguments.F_power
        self.can_comm = arguments.can_comm
        self.patch_num = arguments.patch_num

    def three_dimension_model_reset(self, arguments):
        """3D地图参数的重置，地图的横轴，地图的纵轴，采样的方格的横轴和纵轴"""

        self.L_x = arguments.L_x
        self.L_y = arguments.L_y
        self.S_x = arguments.S_x
        self.S_y = arguments.S_y
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
    def step(self):
        pass

    def render(self):
        pass

if __name__ == "__name__":
    args = arglists()
    new_Env = Env(args)
