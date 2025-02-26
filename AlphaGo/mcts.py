from copy import deepcopy
import time
import numpy as np
import threading
import queue
import torch
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from const import *
from data_loader import to_tensor, to_numpy, pcie_counter


class TreeNode(object):
    def __init__(self, action=None, props=None, parent=None):
        self.parent = parent
        self.action = action
        self.children = []

        self.N = 0  # visit count
        self.Q = .0  # mean action value
        self.W = .0  # total action value
        self.P = props  # prior probability

        self.lock = Lock()  # 添加节点锁用于并发更新

    def is_leaf(self):
        return len(self.children) == 0

    def select_child(self):
        with self.lock:
            index = np.argmax(np.asarray([c.uct() for c in self.children]))
            return self.children[index]

    def uct(self):
        return self.Q + self.P * CPUCT * (np.sqrt(self.parent.N) / (1 + self.N))

    def expand_node(self, props):
        with self.lock:
            self.children = [TreeNode(action=action, props=p, parent=self)
                             for action, p in enumerate(props) if p > 0.]

    def backup(self, v):
        with self.lock:
            self.N += 1
            self.W += v
            self.Q = self.W / self.N


class InferenceTask:
    def __init__(self, state, node, board, state_hash):
        self.state = state
        self.node = node
        self.board = board
        self.state_hash = state_hash


class BatchInferenceWorker:
    def __init__(self, net, batch_size, result_cache):
        self.net = net
        self.batch_size = batch_size
        self.result_cache = result_cache
        self.task_queue = queue.Queue()
        self.cache_lock = Lock()

    def add_task(self, task):
        self.task_queue.put(task)

    def process_batch(self):
        batch = []
        while len(batch) < self.batch_size:
            try:
                task = self.task_queue.get_nowait()
                batch.append(task)
            except queue.Empty:
                break

        if not batch:
            return

        # 构建批量张量并传输到GPU
        states = np.stack([task.state for task in batch], axis=0)
        tensor_states = to_tensor(states, unsqueeze=False)  # 计数器会在这里+1

        # GPU推理结果传回CPU
        with torch.no_grad():
            values, log_props = self.net(tensor_states)
            values = to_numpy(values, USECUDA)      # 计数器会在这里+1
            props = np.exp(to_numpy(log_props, USECUDA))  # 计数器会在这里+1

        # 分发结果
        for idx, task in enumerate(batch):
            value = values[idx][0]
            prop = props[idx].copy()

            # 处理无效动作
            prop[task.board.invalid_moves] = 0.
            total_p = np.sum(prop)
            if total_p > 0:
                prop /= total_p

            # 缓存结果
            with self.cache_lock:
                self.result_cache[task.state_hash] = (value, prop)

            # 节点扩展和反向传播
            if not task.board.is_terminal():
                task.node.expand_node(prop)

            current_node = task.node
            while current_node is not None:
                current_node.backup(-value)
                current_node = current_node.parent

    def run(self):
        while True:
            self.process_batch()
            time.sleep(0.001)  # 避免空转


class MonteCarloTreeSearch(object):
    def __init__(self, net, ms_num=MCTSSIMNUM, batch_size=BATCH_SIZE, num_workers=4):
        self.net = net
        self.ms_num = ms_num
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.result_cache = {}
        self.workers = []
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        # 初始化推理工作线程
        for _ in range(num_workers):
            worker = BatchInferenceWorker(net, batch_size, self.result_cache)
            self.workers.append(worker)
            self.executor.submit(worker.run)

    def search(self, board, node, temperature=.001):
        start_time = time.time()

        self.board = board
        self.root = node
        self.result_cache.clear()

        worker_index = 0
        for _ in range(self.ms_num):
            node = self.root
            board = self.board.clone()

            # Selection
            while not node.is_leaf():
                node = node.select_child()
                board.move(node.action)
                board.trigger()

            # Expansion & Evaluation
            state = board.gen_state()
            state_hash = hash(state.tobytes())

            if state_hash in self.result_cache:
                value, props = self.result_cache[state_hash]

                # Dirichlet噪声（仅根节点）
                if node.parent is None:
                    props = self.dirichlet_noise(props)

                # 归一化概率
                props[board.invalid_moves] = 0.
                total_p = np.sum(props)
                if total_p > 0:
                    props /= total_p

                if not board.is_terminal():
                    node.expand_node(props)

                # 反向传播
                current_node = node
                while current_node is not None:
                    current_node.backup(-value)
                    current_node = current_node.parent
            else:
                # 提交推理任务
                task = InferenceTask(state, node, board.clone(), state_hash)
                self.workers[worker_index].add_task(task)
                worker_index = (worker_index + 1) % self.num_workers

        # 记录总耗时
        elapsed_time = time.time() - start_time
        print(f"搜索耗时: {elapsed_time:.6f} 秒")

        # 决策逻辑
        action_times = np.zeros(self.board.size ** 2)
        for child in self.root.children:
            action_times[child.action] = child.N

        # 过滤无效动作：确保只考虑当前合法的移动
        valid_mask = np.zeros(self.board.size ** 2)
        valid_mask[np.array(self.board.valid_moves)] = 1
        action_times = action_times * valid_mask

        if action_times.sum() == 0:
            # 若没有扩展动作，则均匀选取合法动作
            valid_moves = np.array(self.board.valid_moves)
            pi = np.zeros(self.board.size ** 2, dtype=float)
            pi[valid_moves] = 1.0 / len(valid_moves)
            action = np.random.choice(valid_moves)
        else:
            action, pi = self.decision(action_times, temperature)

        for child in self.root.children:
            if child.action == action:
                return pi, child
        # 如果没有找到匹配的节点，默认创建一个新节点返回
        new_node = TreeNode(action=action)
        return pi, new_node

    @staticmethod
    def dirichlet_noise(props, eps=DLEPS, alpha=DLALPHA):
        return (1 - eps) * props + eps * np.random.dirichlet(np.full(len(props), alpha))

    @staticmethod
    def decision(pi, temperature):
        pi = (1.0 / temperature) * np.log(pi + 1e-10)
        pi = np.exp(pi - np.max(pi))
        pi /= np.sum(pi)
        action = np.random.choice(len(pi), p=pi)
        return action, pi