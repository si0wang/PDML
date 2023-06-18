import random
import numpy as np
from operator import itemgetter

import torch
from torch.utils.data import WeightedRandomSampler
from .tvd import TV_Distance


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

        self.tv_weights = []
        self.device = torch.device("cuda")

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push_batch(self, batch):
        if len(self.buffer) < self.capacity:
            append_len = min(self.capacity - len(self.buffer), len(batch))
            self.buffer.extend([None] * append_len)

        if self.position + len(batch) < self.capacity:
            self.buffer[self.position : self.position + len(batch)] = batch
            self.position += len(batch)
        else:
            self.buffer[self.position : len(self.buffer)] = batch[:len(self.buffer) - self.position]
            self.buffer[:len(batch) - len(self.buffer) + self.position] = batch[len(self.buffer) - self.position:]
            self.position = len(batch) - len(self.buffer) + self.position
            

    def initial_tv_weights(self):
        tv_weight_list = np.ones(len(self.buffer))
        self.tv_weights = tv_weight_list.tolist()
        
    def update_tv_weights(self, tv_list, max_rate):
        tv_weight = (1 / np.array(tv_list[1:-1])) / np.sum(1 / np.array(tv_list[1:-1]))
        warm_up_sample_weight = 0.05 * tv_weight[0]
        newest_sample_weight = (warm_up_sample_weight + np.sum(tv_weight)) * max_rate / (1 - max_rate)
        if newest_sample_weight < np.max(tv_weight):
            newest_sample_weight = np.max(tv_weight)
        tv_weight = np.append(tv_weight, newest_sample_weight)

        tv_weight_list = tv_weight.reshape(len(tv_weight), 1)
        tv_weight_list = np.expand_dims(tv_weight_list, 1).repeat(250, axis=1).flatten()
        warm_up_sample_weight_list = np.array(warm_up_sample_weight).repeat(5000).flatten()
        
        final_weight_list = np.concatenate((warm_up_sample_weight_list, tv_weight_list), axis=0)
        self.tv_weights = final_weight_list.tolist()

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, int(batch_size))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    
    def tvweightedsample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        # batch = random.sample(self.buffer, int(batch_size))
        idx = np.array(list(WeightedRandomSampler(self.tv_weights, batch_size, replacement=True)))
        batch = list(itemgetter(*idx)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done


    def sample_all_batch(self, batch_size):
        idxes = np.random.randint(0, len(self.buffer), batch_size)
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def tvweightedsample_all_batch(self, batch_size):
        # idxes = np.random.randint(0, len(self.buffer), batch_size)
        idxes = np.array(list(WeightedRandomSampler(self.tv_weights, batch_size, replacement=True)))
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done


    def return_all(self):
        state, action, reward, next_state, done = map(np.stack, zip(*self.buffer))
        return state, action, reward, next_state, done
    
    def return_all_(self):
        return self.buffer

    def __len__(self):
        return len(self.buffer)