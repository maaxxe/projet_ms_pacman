"""
@file replay_buffer.py
@brief Buffers de replay pour l'entraînement DQN.
@details Fournit deux implémentations selon le flag USE_PER dans train.py :
- ReplayBuffer : sampling uniforme aléatoire (classique).
- PrioritizedReplayBuffer : sampling priorisé par |TD-error|^alpha (PER).
  Intègre également la fonctionnalité Trophy Buffer (boost par épisode).

PER_EPS est une constante globale partagée pour éviter les priorités nulles.
"""

import random
import numpy as np
from collections import deque

PER_EPS = 1e-6

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, episode_id=None):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s, np.uint8),
            np.array(a, np.int64),
            np.array(r, np.float32),
            np.array(ns, np.uint8),
            np.array(d, np.float32),
        )

    def state_dict(self):
        return {
            'type': 'uniform',
            'capacity': self.capacity,
            'buffer': list(self.buffer),
        }

    def load_state_dict(self, state):
        self.capacity = int(state.get('capacity', self.capacity))
        self.buffer = deque(state.get('buffer', []), maxlen=self.capacity)

    def __len__(self):
        return len(self.buffer)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.alpha = alpha
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        # Tableau circulaire pour stocker l'ID d'épisode de chaque transition
        self.episode_ids = np.full(capacity, -1, dtype=np.int64)
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done, td_error=1.0, episode_id=-1):
        prio = (td_error + PER_EPS) ** self.alpha
        transition = (state, action, reward, next_state, done)
        
        if self.size < self.capacity:
            if len(self.buffer) < self.capacity:
                self.buffer.append(transition)
            else:
                self.buffer[self.pos] = transition
            self.size += 1
        else:
            self.buffer[self.pos] = transition
            
        self.priorities[self.pos] = prio
        self.episode_ids[self.pos] = episode_id if episode_id is not None else -1
        
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        N = len(self)
        if N == 0:
            raise ValueError('Cannot sample from an empty buffer')
        actual_batch = min(batch_size, N)
        prios = self.priorities[:N]
        probs = np.ones(N, dtype=np.float32) / N if prios.sum() <= 0 else prios / prios.sum()
        idxs = np.random.choice(N, actual_batch, p=probs)
        batch = [self.buffer[i] for i in idxs]
        s, a, r, ns, d = zip(*batch)
        s  = np.array(s,  dtype=np.uint8)
        a  = np.array(a,  dtype=np.int64)
        r  = np.array(r,  dtype=np.float32)
        ns = np.array(ns, dtype=np.uint8)
        d  = np.array(d,  dtype=np.float32)
        pt = probs[idxs]
        w = (N * pt) ** (-beta)
        w /= w.max()
        return s, a, r, ns, d, w.astype(np.float32), idxs

    def update_priorities(self, idxs, td_errors):
        prios = (np.abs(td_errors) + PER_EPS) ** self.alpha
        for i, idx in enumerate(idxs):
            self.priorities[idx] = prios[i]

    def boost_episode_priorities(self, episode_id, total_return, baseline, lambda_boost=0.5, scale=100.0):
        """
        @brief  Multiplie la priorité de toutes les transitions appartenant à episode_id.
        @details boost = 1 + lambda_boost * sigmoid((total_return - baseline) / scale)
                 Ne s'applique qu'aux épisodes dont G_t > baseline.
        """
        if episode_id == -1 or total_return <= baseline:
            return None
            
        # Trouver les indices dans le buffer qui correspondent à cet épisode
        valid_idxs = np.where(self.episode_ids[:self.size] == episode_id)[0]
        if len(valid_idxs) == 0:
            return None
            
        # Calcul du boost sigmoid
        x = (total_return - baseline) / scale
        x = max(min(x, 10.0), -10.0) # Eviter l'overflow
        sigmoid = 1.0 / (1.0 + np.exp(-x))
        boost_factor = 1.0 + lambda_boost * sigmoid
        
        # Appliquer le multiplicateur
        self.priorities[valid_idxs] *= boost_factor
        
        return len(valid_idxs), boost_factor

    def state_dict(self):
        return {
            'type': 'per',
            'capacity': self.capacity,
            'alpha': self.alpha,
            'buffer': self.buffer[:self.size],
            'priorities': self.priorities[:self.size].copy(),
            'episode_ids': self.episode_ids[:self.size].copy(),
            'pos': self.pos,
            'size': self.size,
        }

    def load_state_dict(self, state):
        self.capacity = int(state.get('capacity', self.capacity))
        self.alpha = float(state.get('alpha', self.alpha))
        self.buffer = list(state.get('buffer', []))
        self.size = int(state.get('size', len(self.buffer)))
        self.buffer = self.buffer[:self.size]
        
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        saved_prios = np.array(state.get('priorities', []), dtype=np.float32)
        n = min(len(saved_prios), self.size, self.capacity)
        if n > 0:
            self.priorities[:n] = saved_prios[:n]
            
        self.episode_ids = np.full(self.capacity, -1, dtype=np.int64)
        if 'episode_ids' in state:
            saved_ids = np.array(state['episode_ids'], dtype=np.int64)
            n_ids = min(len(saved_ids), self.size, self.capacity)
            if n_ids > 0:
                self.episode_ids[:n_ids] = saved_ids[:n_ids]
                
        self.pos = int(state.get('pos', self.size % max(1, self.capacity))) % max(1, self.capacity)

    def __len__(self):
        return self.size