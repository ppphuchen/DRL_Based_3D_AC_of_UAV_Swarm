from Agent.Agent import Agent

class FUAV(Agent):
    """定义一个FUAV智能体"""
    def __init__(self, arguments):
        """

        :param F_power(float): FUAV对LUAV的默认传输功率
        :param F_comm_radius(int): FUAV的通信范围
        :param F_line_v(int): FUAV的飞行线速度
        ///////////////////////////////////////////////
        """
        super(FUAV, self).__init__()
        #继承了父类的属性
        self.F_power = arguments.F_power
        self.F_comm_radius = arguments.F_comm_radius
        self.F_line_v = arguments.F_line_v
    def reset(self,arguments):
        self.F_power = arguments.F_power
        self.F_comm_radius = arguments.F_comm_radius
        self.F_line_v = arguments.F_line_v
    def step(self):
        pass

if __name__ == "__main__":
    new_FUAV = FUAV()