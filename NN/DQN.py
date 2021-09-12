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
# 超参数
BATCH_SIZE = 32                                 # 样本数量
LR = 0.01                                       # 学习率
EPSILON = 0.9                                   # greedy policy
GAMMA = 0.9                                     # reward discount
TARGET_REPLACE_ITER = 200000                       # 目标网络更新频率
MEMORY_CAPACITY = 200000                          # 记忆库容量
args = arglists()
new_env = Env(args)                             # 创建一个新环境

N_ACTIONS = 4                  # LUAV动作个数 (4个) 向四个方向飞行
N_STATES = 3                    # LUAV状态个数 (3个) LUAV的位置矩阵，覆盖矩阵，连通性矩阵

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

    resize = T.Compose([    # 将经过了不同转换的图像整合到了一起
        T.ToPILImage(),  # 转换成为PILImage
        T.Scale(40, interpolation=Image.CUBIC),  # 缩小或放大
        T.ToTensor()  # 转换成为tensor, (H X W X C) in range(255)=> (C X H X W) in range(1.0)
    ])

class DQN(object):
    def __init__(self):                                                         # 定义DQN的一系列属性
        self.eval_net, self.target_net = Net(), Net()                           # 利用Net创建两个神经网络: 评估网络和目标网络
        self.learn_step_counter = 0                                             # for target updating
        self.memory_counter = 0                                                 # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))             # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.loss_func = nn.MSELoss()                                           # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)

    def choose_action(self, x):  # 定义动作选择函数 (x为状态) (x有三个维度)
        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        #这里的x要改成40*40*1, 100行2列为
        if np.random.uniform() < EPSILON:  # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
            actions_value = self.eval_net.forward(x)  # 通过对评估网络输入状态x，前向传播获得动作值
            action = torch.max(actions_value, 1)[1].data.numpy()  # 输出每一行最大值的索引，并转化为numpy ndarray形式
            action = action[0]  # 输出action的第一个数
        else:  # 随机选择动作
            action = np.random.randint(0, N_ACTIONS)  # 这里action随机等于[0,3]中的一个数 (N_ACTIONS = 4)
        return action  # 返回选择的动作矩阵 [0,3]

    def store_transition(self, s, a, r, s_):  # 定义记忆存储函数 (这里输入为一个transition)
        transition = np.hstack((s, [a, r], s_))  # 在水平方向上拼接数组
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % MEMORY_CAPACITY  # 获取transition要置入的行数
        self.memory[index, :] = transition  # 置入transition
        self.memory_counter += 1  # memory_counter自加1

    def learn(self):                                                            # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:                  # 一开始触发，然后每200000步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())         # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1                                            # 学习步数自加1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)            # 在[0, 2000)内随机抽取32个数，可能会重复
        b_memory = self.memory[sample_index, :]                                 # 抽取32个索引对应的32个transition，存入b_memory
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行4列
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行4列

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.target_net(b_s_).detach()
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_func(q_eval, q_target)
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        loss.backward()                                                 # 误差反向传播, 计算参数更新值
        self.optimizer.step()                                           # 更新评估网络的所有参数



dqn = DQN()                                                             # 令dqn=DQN类

for i in range(800):                                                    # 400个episode循环
    print('<<<<<<<<<Episode: %s' % i)
    s = new_env.reset(args)                                             # 重置环境
    episode_reward_sum = 0                                              # 初始化该循环对应的episode的总奖励

    while True:                                                         # 开始一个episode (每一个循环代表一步)
        new_env.render()                                                # 绘制LUAV的轨迹图
        a = dqn.choose_action(s)                                        # 输入该步对应的状态s，选择动作
        s_, r, done, info = new_env.step(a)                             # 执行动作，获得反馈


        dqn.store_transition(s, a, r, s_)                 # 存储样本
        episode_reward_sum += r                           # 逐步加上一个episode内每个step的reward

        s = s_                                                # 更新状态

        if dqn.memory_counter > MEMORY_CAPACITY:              # 如果累计的transition数量超过了记忆库的固定容量200000
            # 开始学习 (抽取记忆，即32个transition，并对评估网络参数进行更新，并在开始学习后每隔100次将评估网络的参数赋给目标网络)
            dqn.learn()

        if done:       # 如果done为True
            # round()方法返回episode_reward_sum的小数点四舍五入到2个数字
            print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
            break                                             # 该episode结束



