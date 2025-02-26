import numpy as np
import torch
import random

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

from const import *


class PCIeCounter:
    def __init__(self):
        self.cpu_to_gpu = 0
        self.gpu_to_cpu = 0
    
    def reset(self):
        self.cpu_to_gpu = 0
        self.gpu_to_cpu = 0
        
    def log_stats(self):
        print(f"PCIe传输统计:")
        print(f"CPU -> GPU: {self.cpu_to_gpu}次")
        print(f"GPU -> CPU: {self.gpu_to_cpu}次")
        print(f"总传输次数: {self.cpu_to_gpu + self.gpu_to_cpu}次")

pcie_counter = PCIeCounter()

def to_tensor(x, use_cuda=USECUDA, unsqueeze=False):
    if use_cuda:
        pcie_counter.cpu_to_gpu += 1
    if unsqueeze:
        x = x.unsqueeze(0)
    x = torch.from_numpy(x).type(torch.Tensor)
    if use_cuda:
        x = x.cuda()

    if unsqueeze:
        x = x.unsqueeze(0)

    return x


def to_numpy(x, use_cuda=True):
    if use_cuda:
        pcie_counter.gpu_to_cpu += 1
    if use_cuda:
        return x.data.cpu().numpy()
    else:
        return x.data.numpy()


class DataLoader(object):
    def __init__(self, cuda, batch_size):
        self.cuda = cuda
        self.bsz = batch_size

    def __call__(self, datas):
        mini_batch = random.sample(datas, self.bsz)
        states, pi, rewards = [], [], []
        for s, p, r in mini_batch:
            states.append(s)
            pi.append(p)
            rewards.append(r)

        states = to_tensor(np.stack(states, axis=0), use_cuda=self.cuda)
        pi = to_tensor(np.stack(pi, axis=0), use_cuda=self.cuda)
        rewards = to_tensor(np.stack(rewards, axis=0), use_cuda=self.cuda)

        return states, pi, rewards.view(-1, 1)
