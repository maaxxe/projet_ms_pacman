"""
@file      replay_buffer.py
@brief     Buffers de replay pour l'entraînement DQN.
@details   Fournit deux implémentations selon le flag USE_PER dans train.py :
           - ReplayBuffer             : sampling uniforme aléatoire (classique).
           - PrioritizedReplayBuffer  : sampling priorisé par |TD-error|^alpha (PER).

           PER_EPS est une constante globale partagée pour éviter les priorités nulles.
"""

import random
import numpy as np
from collections import deque

## @brief Petit epsilon ajouté aux priorités pour éviter p=0.
#  @details Partagé par PrioritizedReplayBuffer.push() et update_priorities().
PER_EPS = 1e-6

class ReplayBuffer:
    """
    @class   ReplayBuffer
    @brief   Buffer de replay uniforme (utilisé si USE_PER=False dans train.py).
    @details Stocke jusqu'à `capacity` transitions (s, a, r, s', done).
             Le sampling est entièrement aléatoire : chaque transition a
             la même probabilité d'être sélectionnée.
    """

    def __init__(self, capacity):
        """
        @brief  Initialise le buffer avec une deque de taille fixe.
        @param  capacity  Nombre max de transitions stockées (ex: 100_000).
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        @brief  Ajoute une transition au buffer.
        @param  state       Observation actuelle (np.array uint8).
        @param  action      Action choisie (int).
        @param  reward      Récompense reçue (float, clippée si clip_rewards=True).
        @param  next_state  Observation suivante (np.array uint8).
        @param  done        True si épisode terminé.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        @brief  Tire aléatoirement batch_size transitions.
        @param  batch_size  Nombre de transitions à retourner.
        @return Tuple (states, actions, rewards, next_states, dones) en numpy.
        """
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s, np.uint8), np.array(a, np.int64),
                np.array(r, np.float32), np.array(ns, np.uint8),
                np.array(d, np.float32))

    def __len__(self):
        """@brief Retourne le nombre de transitions stockées."""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    @class   PrioritizedReplayBuffer
    @brief   Buffer PER (utilisé si USE_PER=True dans train.py).
    @details Chaque transition reçoit une priorité p = (|TD-error| + PER_EPS)^alpha.
             Le sampling tire proportionnellement à ces priorités :
             P(i) = p_i / sum(p_j). Les transitions "surprenantes" (grande erreur)
             sont révisées plus souvent, ce qui accélère la convergence sur les
             événements rares comme les power pellets de MsPacman.

             Des IS weights (Importance Sampling) w_i = (N * P(i))^(-beta) / max_w
             corrigent le biais introduit par le sampling non-uniforme.
    """

    def __init__(self, capacity, alpha=0.6):
        """
        @brief  Initialise le PER buffer.
        @param  capacity  Taille max du buffer (ex: 100_000).
        @param  alpha     Exposant de priorité (PER_ALPHA=0.6).
                          0 = uniforme, 1 = full greedy par TD-error.
        """
        self.alpha = alpha
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros(capacity)  # Tableau circulaire de priorités
        self.pos = 0                          # Position d'écriture courante

    def push(self, state, action, reward, next_state, done, td_error=1.0):
        """
        @brief  Ajoute une transition avec priorité initiale td_error^alpha.
        @param  state       Observation actuelle.
        @param  action      Action choisie.
        @param  reward      Récompense reçue.
        @param  next_state  Observation suivante.
        @param  done        True si épisode terminé.
        @param  td_error    Erreur TD initiale (1.0 par défaut = haute prio).
        @details Les nouvelles transitions ont td_error=1.0 pour être vues au
                 moins une fois avant d'être repriorisées par update_priorities().
        """
        prio = (td_error + PER_EPS) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        @brief  Sample priorisé + calcul IS weights.
        @param  batch_size  Nombre de transitions à retourner.
        @param  beta        Exposant IS weight (PER_BETA_START=0.4 → 1.0 en fin).
                            beta=0 : pas de correction, beta=1 : correction totale.
        @return Tuple (states, actions, rewards, next_states, dones, weights, idxs).
                weights et idxs sont nécessaires pour appeler update_priorities().
        """
        N = len(self)
        if N <= batch_size:
            batch = [self.buffer[i] for i in range(N)]
            s, a, r, ns, d = zip(*batch)
            return (np.array(s), np.array(a), np.array(r),
                    np.array(ns), np.array(d), np.ones(N), np.arange(N))

        prios = self.priorities[:N]
        probs = prios / prios.sum()
        idxs = np.random.choice(N, batch_size, p=probs)

        batch = [self.buffer[i] for i in idxs]
        s, a, r, ns, d = zip(*batch)
        s, a, r, ns, d = map(np.array, [s, a, r, ns, d])

        # IS weights pour corriger le biais de sampling non-uniforme
        pt = prios[idxs] / prios.sum()
        w = (N * pt) ** (-beta)
        w /= w.max()   # Normalisation pour que max_w = 1

        return s, a, r, ns, d, w, idxs

    def update_priorities(self, idxs, td_errors):
        """
        @brief  Met à jour les priorités après calcul des nouvelles TD-errors.
        @param  idxs      Indices des transitions échantillonnées.
        @param  td_errors Nouvelles erreurs TD (np.array float).
        @details Appelé après chaque step d'optimisation dans train.py
                 uniquement si USE_PER=True.
        """
        prios = (np.abs(td_errors) + PER_EPS) ** self.alpha
        for i, idx in enumerate(idxs):
            self.priorities[idx] = prios[i]

    def __len__(self):
        """@brief Retourne le nombre de transitions stockées."""
        return min(self.pos, self.capacity)
