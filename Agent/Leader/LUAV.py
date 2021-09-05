from Agent.Agent import Agent

class LUAV(Agent):
    """定义一个LUAV智能体"""

    def __init__(self, arguments):
        """
        :param patch_fuav_num(int) 子群中FUAV的数量
        :param can_comm: (float) 环境中LUAV之间能够通信的最小功率
        :param patch_num: (int) 环境中子群的数量
        :param L_power: (float) 环境中LUAV的发送功率
        :param L_UAV_num: (int) 环境中LUAV的数量
        :param L_comm_radius: (int) L_UAV的通信范围
        :param gain_r: (float) LUAV的接收增益
        :param gain_t: (float) LUAV的发送增益
        :param re_cover_rate: (float) LUAV的重复覆盖率
        :param average_collect_time: (int) 子群飞行每米后所需要的停下来收集数据的时间
        ////////////////////////////////////////////////
        :param max_sub_group_cover_time: (int) 所有的子群总覆盖时间中的最大值
        """
        super(LUAV, self).__init__()
        # 继承了父类的属性
        self.patch_fuav_num = arguments.patch_fuav_num
        self.can_comm = arguments.can_comm
        self.patch_num = arguments.patch_num
        self.L_power = arguments.L_power
        self.L_UAV_num = arguments.L_UAV_num
        self.L_comm_radius = arguments.L_comm_radius
        self.gain_r = arguments.gain_r
        self.gain_t = arguments.gain_t
        self.re_cover_rate = arguments.re_cover_rate
        self.average_collect_time = arguments.average_collect_time
        self.max_sub_group_cover_time = arguments.max_sub_group_cover_time

    def reset(self, arguments):
        """重置LUAV的参数"""
        self.patch_fuav_num = arguments.patch_fuav_num
        self.can_comm = arguments.can_comm
        self.patch_num = arguments.patch_num
        self.L_power = arguments.L_power
        self.L_UAV_num = arguments.L_UAV_num
        self.L_comm_radius = arguments.L_comm_radius
        self.gain_r = arguments.gain_r
        self.gain_t = arguments.gain_t
        self.re_cover_rate = arguments.re_cover_rate
        self.average_collect_time = arguments.average_collect_time
        self.max_sub_group_cover_time = arguments.max_sub_group_cover_time

    def step(self):
        pass

if __name__ == "__main__":
    new_LUAV = LUAV()