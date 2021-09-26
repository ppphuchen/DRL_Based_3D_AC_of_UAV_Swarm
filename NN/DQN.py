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
from collections import namedtuple
from itertools import count
from PIL import Image
from Env.Env import Env
from Arguments.args import arglists
from Function.ToolFunction import calculate_distance
from Function.ToolFunction import dfs
from Function.ToolFunction import judge_done
# 超参数
Berta_s = 1000                                   # 覆盖率奖励的惩罚因子
Berta_c = 100                                   # 连通性的惩罚因子
Berta_t = 10                                   # 消耗一个时间步的惩罚因子
Berta_pv = 10000                                  # LUAV重复覆盖的惩罚因子
Boundary_argument = 1.0                         # 每个LUAV的初始边界范围的比例

BATCH_SIZE = 32                                 # 每次采样的样本数量
LR = 0.01                                       # 学习率
EPSILON = 0.01                                   # greedy policy
GAMMA = 1                                     # reward discount
TARGET_REPLACE_ITER = 200000                       # 目标网络更新频率
MEMORY_CAPACITY = 200000                          # 记忆库容量
args = arglists()
new_env = Env(args)                             # 创建一个新环境

N_ACTIONS = 4                  # LUAV动作个数 (4个) 向四个方向飞行
N_STATES = 104                    # LUAV状态个数 (3个) LUAV的位置矩阵，覆盖矩阵，连通性矩阵

"""
torch.nn是专门为神经网络设计的模块化接口。nn构建于Autograd之上，可以用来定义和运行神经网络。
nn.Module是nn中十分重要的类，包含网络各层的定义及forward方法。
定义网络：
    需要继承nn.Module类，并实现forward方法。
    一般把网络中具有可学习参数的层放在构造函数__init__()中。
    只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现(利用Autograd)。
"""


class Net(nn.Module):
    def __init__(self):             # 定义Net的一系列属性
        super(Net, self).__init__()

        self.conv1 = nn.Sequential( #input_size = 40*40*1
            nn.Conv2d(1, 16, kernel_size=5),
            #卷积层的输入通道数等于输入图像的输入通道数，因为卷积核会把每个通道卷积后的值按通道相加，
            # 所以一个卷积核只会输出一个通道的值
            # 输出图像通道数就等于卷积核的数量
            #out_size = 36*36*16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=17, stride=1),
            #池化的时候，池化层对输入图像的每个通道分别池化，而不会按通道相加，所以经过池化层的输入通道数等于输出通道数
        )#out_size = 20*20*16

        self.conv2 = nn.Sequential( #input_size = 20*20*16
            nn.Conv2d(16, 32, kernel_size=3),
            #out_size = 18*18*32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )#out_size = 10*10*32

        self.fc = nn.Linear(3200, 100)  #完全连接层

    def forward(self, x):
        conv1 = self.conv1(x)    #第一层卷积的输出
        conv2 = self.conv2(conv1)#第二层卷积的输出
        fc_input = conv2.view(conv2.size(0), -1)  #将一个多行的tensor变成一行
        fc_out = self.fc(fc_input) #完全连接层的输出
        return fc_out
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Sequential( # input_size = 1*104
            nn.Linear(104, 100),
            nn.ReLU(),
        )#out_size = 1*100
        self.conv2 = nn.Sequential( #input_size = 1*100
            nn.Linear(100, 10),
            nn.ReLU(),
        )#out_size = 1*10
        self.conv3 = nn.Sequential( #input_size = 1*10
            nn.Linear(10, 4),
            nn.Softmax(dim=0),
        )
    def forward(self, x):
        conv1 = self.conv1(x) #第一个隐藏层的输出
        conv2 = self.conv2(conv1) #第一个隐藏层的输出
        conv3 = self.conv3(conv2) # 第三个隐藏层的输出
        return conv3
class DQN(object):
    def __init__(self):                                                         # 定义DQN的一系列属性
        self.eval_net, self.target_net = Net(), Net()                           # 利用Net创建两个神经网络: 评估网络和目标网络
        self.q_eval_net, self.q_target_net = Net2(), Net2()                      # 第二层的NN，用来输出action
        self.learn_step_counter = 0                                             # for target updating
        self.memory_counter = 0                                                 # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, 2091))                          # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.loss_func = nn.MSELoss()                                           # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
        self.observation = []
        self.action = []
    def get_observation(self):
        """

        :param map_cover: 40*40的覆盖矩阵
        :return: 1*104 的张量数组，数组的每个元素代表了第i个LUAV的   （1*2）
        """
        self.observation = []
        self.action = []
        self.map_cover = new_env.map_cover            #得到此时状态的map_cover
        self.input_cover = torch.FloatTensor(new_env.map_cover.reshape((1,1,40,40)))
        self.history_cover = self.eval_net.forward(self.input_cover)      #输出为 1*100的向量
        for i in range(10):
            self.luav_pos_sum = torch.tensor([0.0, 0.0])
            self.luav_nei_pos_sum = torch.tensor([0.0,0.0])
            self.luav_nie_history_pos = torch.tensor([0.0, 0.0])
            self.luav_history_pos = torch.tensor([0.0, 0.0])
            if(i>0 and i<9):

                for j in range( len(new_env.luav_history_pos[i-1]) ):
                    self.luav_nei_pos_sum = self.luav_nei_pos_sum + new_env.luav_history_pos[i-1][j] # 加上左邻居的历史位置

                for j in range(len(new_env.luav_history_pos[i+1])):
                    self.luav_nei_pos_sum = self.luav_nei_pos_sum + new_env.luav_history_pos[i+1][j] # 加上右邻居的位置

                self.luav_nie_history_pos = self.luav_nei_pos_sum / (len(new_env.luav_history_pos[i-1])+len(new_env.luav_history_pos[i+1]))

            elif(i==0):
                for j in range(len(new_env.luav_history_pos[i+1])):
                    self.luav_nei_pos_sum = self.luav_nei_pos_sum + new_env.luav_history_pos[i+1][j] #只有右邻居的位置
                self.luav_nie_history_pos = self.luav_nei_pos_sum / len(new_env.luav_history_pos[i+1])

            else:
                for j in range(len(new_env.luav_history_pos[i-1])):
                    self.luav_nei_pos_sum = self.luav_nei_pos_sum + new_env.luav_history_pos[i-1][j] #只有左邻居的位置
                self.luav_nie_history_pos = self.luav_nei_pos_sum / len(new_env.luav_history_pos[i-1])
            #luav的邻居的历史平均位置

            for j in range(len(new_env.luav_history_pos[i])):
               self.luav_pos_sum += new_env.luav_history_pos[i][j]
            self.luav_history_pos = self.luav_pos_sum/len(new_env.luav_history_pos[i])  #第i个luav的pos

            self.luav_history_pos = torch.FloatTensor(self.luav_history_pos.reshape(1, 2))
            self.luav_nie_history_pos = torch.FloatTensor(self.luav_nie_history_pos.reshape(1, 2))
            self.observation.append(torch.hstack((self.luav_history_pos, self.history_cover, self.luav_nie_history_pos)))
            if np.random.uniform(0, 1) < EPSILON:  #做一个greedy
                 self.action.append(int(torch.max(self.q_eval_net(self.observation[i]), dim=1)[1]))
            else:
                 self.action.append(np.random.randint(0, 4))
    def execute_action(self):
        self.done = 0
        for i in range(10):
            #第i个LUAV的边界
            left_bound = i*4*25
            right_bound = ((i+1)*4-1)*25
            up_bound = 39*25
            down_bound = 0
            old_pos_x = new_env.L_UAV_pos[i][0]
            old_pos_y = new_env.L_UAV_pos[i][1]
            #实际位置更新,LUAV的格子位置更新,实际环境的map_cover更新
            if(self.action[i]==0):
                if(new_env.L_UAV_pos[i][1]+25<=up_bound):
                    new_env.L_UAV_pos[i][1] = new_env.L_UAV_pos[i][1]+25
                    new_env.luav_point_pos[i][1] = new_env.luav_point_pos[i][1]+1
                    new_env.map_cover[int(new_env.luav_point_pos[i][0])][int(new_env.luav_point_pos[i][1])] = 1
                    print(1)
            if(self.action[i]==1):
                if(new_env.L_UAV_pos[i][1]>=30*25):  #突破原有的划分通道的束缚
                    right_bound = 39*25
                if(new_env.L_UAV_pos[i][0]+25<=right_bound):
                    new_env.L_UAV_pos[i][0] = new_env.L_UAV_pos[i][0]+25
                    new_env.luav_point_pos[i][0] = new_env.luav_point_pos[i][0]+1
                    new_env.map_cover[int(new_env.luav_point_pos[i][0])][int(new_env.luav_point_pos[i][1])] = 1
            if(self.action[i]==2):
                if(new_env.L_UAV_pos[i][1]-25>=down_bound):
                    new_env.L_UAV_pos[i][1] = new_env.L_UAV_pos[i][1]-25
                    new_env.luav_point_pos[i][1] = new_env.luav_point_pos[i][1] - 1
                    new_env.map_cover[int(new_env.luav_point_pos[i][0])][int(new_env.luav_point_pos[i][1])] = 1
            if(self.action[i]==3):
                if(new_env.L_UAV_pos[i][1]>=30*25):
                    left_bound = 0
                if(new_env.L_UAV_pos[i][0]-25>=left_bound):
                    new_env.L_UAV_pos[i][0] = new_env.L_UAV_pos[i][0]-25
                    new_env.luav_point_pos[i][0] = new_env.luav_point_pos[i][0] - 1
                    new_env.map_cover[int(new_env.luav_point_pos[i][0])][int(new_env.luav_point_pos[i][1])] = 1
            if(old_pos_x!=new_env.L_UAV_pos[i][0] or old_pos_y!=new_env.L_UAV_pos[i][1]): #判断LUAV是否进行了移动
                #如果移动则记录位置，更新行数
                new_env.render_row[i] = new_env.render_row[i]+1
                new_env.render_pos[int(new_env.render_row[i])][i*2] = new_env.L_UAV_pos[i][0]
                new_env.render_pos[int(new_env.render_row[i])][i*2+1] = new_env.L_UAV_pos[i][1]
            #历史位置更新，自动丢弃旧的位置
            new_env.luav_history_pos[i].append(torch.tensor([new_env.L_UAV_pos[i][0], new_env.L_UAV_pos[i][1]]))
            #由新得到的环境中的map_cover,计算done

        self.done = judge_done(new_env.map_cover)

    def get_reward(self):
        """
        :param r_ac: action后地图的总体覆盖率
        :param f_ct: action后的LUAV的连通性
        :param self.map_cover: action之前的地图覆盖情况
        计算LUAV之间的共享奖励reward_share，每个LUAV的个体奖励reward_one，以及将前面两者累加起来的共同奖励reward_common
        action后的新map_cover, 由action后新的LUAV的位置计算得到连通性矩阵
        :return:
        """
        self.reward_one = np.zeros(10)
        cover_counter = 0
        for i in range(10):
            for j in range(10):
                if(new_env.map_cover[i][j]==1):
                    cover_counter = cover_counter+1
        r_ac = cover_counter/(1600)
        f_ct = dfs(calculate_distance(new_env.L_UAV_pos))
        self.reward_share = Berta_s*r_ac - Berta_c*f_ct - Berta_t
        reward_sum = 0
        for i in range(10):
            if self.map_cover[int(new_env.luav_point_pos[i][0])][int(new_env.luav_point_pos[i][1])] == 1:
                self.reward_one[i] = -Berta_pv
            else:
                self.reward_one[i] = 0
                self.map_cover[int(new_env.luav_point_pos[i][0])][int(new_env.luav_point_pos[i][1])] = 1
            reward_sum = reward_sum + self.reward_one[i]

        self.reward_common = self.reward_share + reward_sum
        #更新map_cover
        self.map_cover = new_env.map_cover
    def get_next_observation(self):
        """
        :param map_cover:更新后的40*40 的覆盖矩阵
        :return:经过action后的下一个observation
        """
        self.next_observation = []
        self.input_cover = torch.FloatTensor(self.map_cover.reshape((1, 1, 40, 40)))
        self.history_cover = self.eval_net.forward(self.input_cover)  # 输出为 1*100的向量
        for i in range(10):
            self.luav_pos_sum = torch.tensor([0.0, 0.0])
            self.luav_nei_pos_sum = torch.tensor([0.0,0.0])
            self.luav_nie_history_pos = torch.tensor([0.0, 0.0])
            self.luav_history_pos = torch.tensor([0.0, 0.0])
            if(i>0 and i<9):

                for j in range( len(new_env.luav_history_pos[i-1]) ):
                    self.luav_nei_pos_sum = self.luav_nei_pos_sum + new_env.luav_history_pos[i-1][j] # 加上左邻居的历史位置

                for j in range(len(new_env.luav_history_pos[i+1])):
                    self.luav_nei_pos_sum = self.luav_nei_pos_sum + new_env.luav_history_pos[i+1][j] # 加上右邻居的位置

                self.luav_nie_history_pos = self.luav_nei_pos_sum / (len(new_env.luav_history_pos[i-1])+len(new_env.luav_history_pos[i+1]))

            elif(i==0):
                for j in range(len(new_env.luav_history_pos[i+1])):
                    self.luav_nei_pos_sum = self.luav_nei_pos_sum + new_env.luav_history_pos[i+1][j] #只有右邻居的位置
                self.luav_nie_history_pos = self.luav_nei_pos_sum / len(new_env.luav_history_pos[i+1])

            else:
                for j in range(len(new_env.luav_history_pos[i-1])):
                    self.luav_nei_pos_sum = self.luav_nei_pos_sum + new_env.luav_history_pos[i-1][j] #只有左邻居的位置
                self.luav_nie_history_pos = self.luav_nei_pos_sum / len(new_env.luav_history_pos[i-1])
            # luav的邻居的历史平均位置

            for j in range(len(new_env.luav_history_pos[i])):
                self.luav_pos_sum += new_env.luav_history_pos[i][j]
            self.luav_history_pos = self.luav_pos_sum / len(new_env.luav_history_pos[i])  # 第i个luav的pos
            self.luav_history_pos = torch.FloatTensor(self.luav_history_pos.reshape(1, 2))
            self.luav_nie_history_pos = torch.FloatTensor(self.luav_nie_history_pos.reshape(1, 2))
            self.next_observation.append(
                torch.hstack((self.luav_history_pos, self.history_cover, self.luav_nie_history_pos)))

    def store_transition(self, observation, action, next_observation, reward_common):
        """

        :param observation: 该step执行前的 observation
        :param action: 该step下的action
        :param next_observation: 该step下执行action后的next_observation
        :param reward_common: 该step下执行后的共同奖励
        :return:
        """
        transition = []  # s[i],a[i],next_s[i]..... reward_common
        list = [action[0]]
        x = np.array(list)
        transition = np.hstack((observation[0].detach().numpy() , np.expand_dims(x, 1), next_observation[0].detach().numpy()))
        for i in range(1, 10):
            list = [action[0]]
            x = np.array(list)
            transition = np.hstack((transition, observation[i].detach().numpy(), np.expand_dims(x,1), next_observation[i].detach().numpy()))
        list = [reward_common]
        x = np.array(list)
        transition = np.hstack((transition, np.expand_dims(x, 1)))
        index = self.memory_counter % MEMORY_CAPACITY    #获取transition要置入的行数
        self.memory[index,:] = transition                #放入transition
        self.memory_counter = self.memory_counter + 1    #样本计数器加1

    def learn(self):
        # 目标参数网络更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())         #将评估网络的参数赋给目标网络
        self.learn_step_counter = self.memory_counter + 1                       #学习步数加1

        # 抽取buffer里的数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)            #200000 , 32, 随机抽取32个数字并组成一个数组
        batch_memrory = self.memory[sample_index, :]                            #抽取其中的32行
        batch_memrory_s = np.zeros(10)
        batch_memrory_a = np.zeros(10)
        batch_memrory_next_s = np.zeros(10)
        for i in range(10):
            batch_memrory_s.append(torch.FloatTensor(batch_memrory[:, i*(N_STATES*2+1):i*(N_STATES*2+1)+N_STATES]) )      #抽出 32*104 为状态值
            batch_memrory_a.append(torch.LongTensor(batch_memrory[:, i*(N_STATES*2+1)+N_STATES]))    #动作值
            batch_memrory_next_s.append(torch.FloatTensor(batch_memrory[:, i*(N_STATES*2+1)+N_STATES+1 :i*(N_STATES*2+1)+N_STATES*2+1 ]))  #下一个状态值
        batch_memrory_r = batch_memrory[:, 2090]   #共同奖励
        q_eval_sum = 0
        q_next_sum = 0
        for i in range(10):
            q_eval_sum = q_eval_sum + self.q_eval_net(batch_memrory_s[i]).gather(1, batch_memrory_a[i])
            # 通过评估网络求出32行状态对应的action值，然后对每行对应索引的q值进行聚合
            q_next = self.q_target_net(batch_memrory_next_s[i]).detach()
            q_next_sum = q_next_sum + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        q_target = batch_memrory_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval_sum, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()
for i in range(800):
    print('<<<<<<<<<Episode: %s' % i)
    new_env.reset(args)     # 重置环境
    episode_reward_sum = 0  # 初始化该循环对应的episode的总奖励
    while True:
        if dqn.memory_counter % 5000 == 0:
            new_env.render()  # 绘制轨迹图
        dqn.get_observation()                                                  #得到动作值
        dqn.execute_action()                                                   #执行动作更新状态
        dqn.get_reward()                                                       #更新奖励
        dqn.get_next_observation()                                             #获取下一个状态
        dqn.store_transition(dqn.observation, dqn.action, dqn.next_observation, dqn.reward_common)  #存储样本
        episode_reward_sum = episode_reward_sum + dqn.reward_common            #逐步加上每个step的奖励

        if dqn.memory_counter > MEMORY_CAPACITY:                               #如果累积的样本大于样本容量则开始学习
            dqn.learn()
        if dqn.done:
            break
