from collections import namedtuple
import random
import torch
import os
import pickle

Transition = namedtuple('Transition',
                        ('current_sequence',
                         'current_protein_sequence',
                         'next_sequence',
                         'next_protein_sequence',
                         'reward',
                         'degradation_reward',
                         'protein_mask'))


class ReplayMemory(object):

    def __init__(self, capacity, memory_folder='memory'):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.memory_folder=memory_folder
        os.system(f'mkdir {memory_folder}')

    def push(self, *args):
        """Saves a transition."""
        #if len(self.memory) < self.capacity:
        # self.memory.append(None)
        # self.memory[-1] = Transition(*args)
        # #self.position = (self.position + 1) % self.capacity
        # if len(self.memory) > self.capacity:
        #     self.memory=self.memory[-self.capacity:]
        memory=Transition(*args)
        with open(f'{self.memory_folder}/transition{self.position}.p','wb+') as f:
            pickle.dump(memory,f)
        self.position+=1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
