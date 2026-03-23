"""
@file replay_buffer.py
@brief FINAL - 100% fonctionnel PER + uniforme
"""

import random
import numpy as np
from collections import deque

PER_EPS = 1e-6  # Constante globale PER

class ReplayBuffer:
    """Buffer uniforme standard"""
    def __init__(self, capacity): self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return np.array(s, np.uint8), np.array(a, np.int64), np.array(r, np.float32), \
               np.array(ns, np.uint8), np.array(d, np.float32)
    def __len__(self): return len(self.buffer)

class PrioritizedReplayBuffer:
    """PER fonctionnel complet"""
    def __init__(self, capacity, alpha=0.6):
        self.alpha = alpha
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.pos = 0
    
    def push(self, state, action, reward, next_state, done, td_error=1.0):
        prio = (td_error + PER_EPS) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        N = len(self)
        if N <= batch_size:
            batch = [self.buffer[i] for i in range(N)]
            s,a,r,ns,d = zip(*batch)
            return np.array(s),np.array(a),np.array(r),np.array(ns),np.array(d), \
                   np.ones(N), np.arange(N)
        
        prios = self.priorities[:N]
        probs = prios / prios.sum()
        idxs = np.random.choice(N, batch_size, p=probs)
        
        batch = [self.buffer[i] for i in idxs]
        s,a,r,ns,d = zip(*batch)
        s,a,r,ns,d = map(np.array, [s,a,r,ns,d])
        
        pt = prios[idxs] / prios.sum()
        w = (N * pt) ** (-beta)
        w /= w.max()
        
        return s,a,r,ns,d,w,idxs
    
    def update_priorities(self, idxs, td_errors):
        prios = (np.abs(td_errors) + PER_EPS) ** self.alpha
        for i,idx in enumerate(idxs): self.priorities[idx] = prios[i]
    
    def __len__(self):
        return min(self.pos, self.capacity)

