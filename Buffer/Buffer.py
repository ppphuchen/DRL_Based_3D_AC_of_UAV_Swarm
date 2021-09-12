import random
import numpy as np
import collections
Transition = collections.namedtuple('Transition', 'state', 'action', 'reward','observation', 'next_state')
class replay_memorary(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push_date(self, *args):
        """
        存一次样本
        *args 作为形参可以将多个元素整合成一个tuple
        **kwargs 作为形参可以把有关键字的值整合成一个字典
        """
        if self.memory < self.capacity:
            self.memory.append(None)
        #加一个空位

        self.memory[self.position] = Transition(args)
        #形成一个命名元组，然后存到当前位置，可用 self.memory[pos][name] / self.memory[pos1][pos2]访问
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        """
        :param batch_size: 用于截取的样本长度
        :return: 返回一个随机抽取但是抽取后的元素顺序与原顺序保持一致的list
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


