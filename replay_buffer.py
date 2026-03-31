"""
@file      replay_buffer.py
@brief     Buffers de replay pour l'entraînement DQN.
@details   Fournit trois implémentations selon les flags dans train.py :
           - ReplayBuffer                : sampling uniforme aléatoire (classique).
           - PrioritizedReplayBuffer     : sampling priorisé par |TD-error|^alpha (PER).
           - PrioritizedReplayBuffer     : avec extension Trophy Buffer si USE_TROPHY_BUFFER=True.

           ### Trophy Buffer (idée originale)
           En fin d'épisode, toutes les transitions de cet épisode voient leur
           priorité **boostée proportionnellement au return total G_t** de l'épisode,
           comparé à une baseline (moyenne mobile des returns récents) :

           @code
           boost(G_t) = 1 + λ · sigmoid((G_t - baseline) / scale)
           new_prio(i) = old_prio(i) · boost(G_t)
           @endcode

           Résultat : les transitions issues d'épisodes **exceptionnels** sont
           rejouées bien plus souvent que celles d'épisodes médiocres, même si
           leur TD-error locale est faible (ex: une décision anodine qui précède
           un enchaînement brillant). Fusion conceptuelle entre Self-Imitation
           Learning (Oh et al. 2018) et PER appliquée rétroactivement sur les
           priorités.

           PER_EPS est une constante globale partagée pour éviter les priorités nulles.
"""

import random
import numpy as np
from collections import deque, defaultdict

## @brief Petit epsilon ajouté aux priorités pour éviter p=0.
#  @details Partagé par PrioritizedReplayBuffer.push() et update_priorities().
PER_EPS = 1e-6

## @brief Priorité maximale autorisée (évite les explosions numériques).
#  @details Cap appliqué dans boost_episode_priorities() après multiplication.
MAX_PRIORITY = 1e6


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

    def push(self, state, action, reward, next_state, done, **kwargs):
        """
        @brief  Ajoute une transition au buffer.
        @param  state       Observation actuelle (np.array uint8).
        @param  action      Action choisie (int).
        @param  reward      Récompense reçue (float, clippée si clip_rewards=True).
        @param  next_state  Observation suivante (np.array uint8).
        @param  done        True si épisode terminé.
        @param  kwargs      Paramètres ignorés (episode_id, td_error) pour compatibilité.
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
    @brief   Buffer PER avec support optionnel du Trophy Buffer.
    @details
    ### Mode PER seul (USE_TROPHY_BUFFER=False)
    Chaque transition reçoit une priorité p = (|TD-error| + PER_EPS)^alpha.
    Le sampling tire proportionnellement à ces priorités :
    P(i) = p_i / sum(p_j). Les transitions "surprenantes" (grande erreur)
    sont révisées plus souvent, ce qui accélère la convergence sur les
    événements rares comme les power pellets de MsPacman.
    Des IS weights (Importance Sampling) w_i = (N * P(i))^(-beta) / max_w
    corrigent le biais introduit par le sampling non-uniforme.

    ### Mode Trophy Buffer (USE_TROPHY_BUFFER=True)
    En plus du PER standard, le buffer maintient un index épisode → positions.
    À la fin de chaque épisode, train.py appelle boost_episode_priorities()
    qui multiplie toutes les priorités de cet épisode par :
    @code
    boost = 1 + λ · sigmoid((G_t - baseline) / scale)
    @endcode
    Cela revalorise rétroactivement les transitions d'épisodes exceptionnels,
    même celles dont la TD-error locale était faible au moment du push.
    """

    def __init__(self, capacity, alpha=0.6):
        """
        @brief  Initialise le PER buffer avec structures Trophy si nécessaire.
        @param  capacity  Taille max du buffer (ex: 100_000).
        @param  alpha     Exposant de priorité (PER_ALPHA=0.6).
                          0 = uniforme, 1 = full greedy par TD-error.
        """
        self.alpha     = alpha
        self.capacity  = capacity
        self.buffer    = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos       = 0  ##< Position d'écriture courante (buffer circulaire).

        # --- Structures Trophy Buffer ---
        ## @brief Mapping position → episode_id courant à cet emplacement.
        #  @details Initialisé à -1 (aucun épisode). Mis à jour à chaque push().
        #           Permet de nettoyer episode_positions quand une case est écrasée.
        self._ep_of_pos = np.full(capacity, -1, dtype=np.int64)

        ## @brief Mapping episode_id → liste de positions dans le buffer circulaire.
        #  @details Rempli par push() si episode_id fourni. Consommé et supprimé
        #           par boost_episode_priorities() après le boost.
        self._ep_positions = defaultdict(list)

    # -------------------------------------------------------------------------
    # Interface principale
    # -------------------------------------------------------------------------

    def push(self, state, action, reward, next_state, done,
             td_error=1.0, episode_id=None):
        """
        @brief  Ajoute une transition avec priorité initiale et tracking épisode.
        @param  state       Observation actuelle.
        @param  action      Action choisie.
        @param  reward      Récompense reçue.
        @param  next_state  Observation suivante.
        @param  done        True si épisode terminé.
        @param  td_error    Erreur TD initiale (1.0 par défaut = haute prio).
        @param  episode_id  Identifiant de l'épisode courant (int ou None).
                            Requis pour le Trophy Buffer. Si None, le tracking
                            épisode est désactivé pour cette transition.
        @details
        1. Nettoie l'ancien épisode mappé à cette position (buffer circulaire).
        2. Stocke la transition.
        3. Enregistre la position dans _ep_positions[episode_id] si fourni.
        """
        pos = self.pos
        prio = (abs(td_error) + PER_EPS) ** self.alpha

        # Nettoyage de l'ancienne entrée à cette position (buffer circulaire)
        old_ep = int(self._ep_of_pos[pos])
        if old_ep >= 0 and old_ep in self._ep_positions:
            try:
                self._ep_positions[old_ep].remove(pos)
            except ValueError:
                pass
            if not self._ep_positions[old_ep]:
                del self._ep_positions[old_ep]

        # Stockage de la transition
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[pos] = (state, action, reward, next_state, done)
        self.priorities[pos] = prio

        # Tracking épisode pour Trophy Buffer
        if episode_id is not None:
            self._ep_of_pos[pos] = episode_id
            self._ep_positions[episode_id].append(pos)
        else:
            self._ep_of_pos[pos] = -1

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
            return (np.array(s), np.array(a, np.int64), np.array(r, np.float32),
                    np.array(ns), np.array(d, np.float32),
                    np.ones(N, dtype=np.float32), np.arange(N))

        prios = self.priorities[:N]
        probs = prios / prios.sum()
        idxs  = np.random.choice(N, batch_size, replace=False, p=probs)

        batch  = [self.buffer[i] for i in idxs]
        s, a, r, ns, d = zip(*batch)

        # IS weights : corrigent le biais de sampling non-uniforme
        pt = prios[idxs] / prios.sum()
        w  = (N * pt) ** (-beta)
        w /= w.max()  # Normalisation pour que max_w = 1

        return (np.array(s, np.uint8), np.array(a, np.int64),
                np.array(r, np.float32), np.array(ns, np.uint8),
                np.array(d, np.float32), w.astype(np.float32), idxs)

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
            self.priorities[idx] = float(prios[i])

    # -------------------------------------------------------------------------
    # Trophy Buffer
    # -------------------------------------------------------------------------

    def boost_episode_priorities(self, episode_id, total_return,
                                  baseline, lambda_boost=0.5):
        """
        @brief  Booste rétroactivement les priorités de toutes les transitions
                d'un épisode terminé, proportionnellement à son return total.
        @param  episode_id    Identifiant de l'épisode (int, identique à celui
                              passé à push() durant l'épisode).
        @param  total_return  Return total G_t de l'épisode (score brut ou clippé
                              selon la configuration utilisée).
        @param  baseline      Référence de comparaison (moyenne mobile des returns
                              récents), fournie par train.py. Permet de centrer
                              le sigmoid et d'adapter l'échelle automatiquement.
        @param  lambda_boost  Intensité maximale du boost (défaut 0.5).
                              boost ∈ [1.0, 1 + lambda_boost].
                              - Si G_t >> baseline : boost ≈ 1 + lambda_boost (max).
                              - Si G_t == baseline : boost ≈ 1 + lambda_boost / 2.
                              - Si G_t << baseline : boost ≈ 1.0 (quasi-inchangé).
        @details
        Formule appliquée :
        @code
        scale  = max(|baseline|, 100.0)          # évite division par zéro
        boost  = 1 + λ · sigmoid((G - baseline) / scale)
        new_p  = clip(old_p · boost, PER_EPS, MAX_PRIORITY)
        @endcode
        Les positions invalides (écrasées par le buffer circulaire entre le
        push et le boost) sont ignorées silencieusement.
        Après boost, l'entrée episode_id est supprimée de _ep_positions
        pour libérer la mémoire.
        """
        if episode_id not in self._ep_positions:
            return  # Épisode déjà consommé ou inconnu

        positions = self._ep_positions[episode_id]
        if not positions:
            del self._ep_positions[episode_id]
            return

        # Calcul du facteur de boost via sigmoid centré sur la baseline
        scale = max(abs(baseline), 100.0)
        sigmoid_val = 1.0 / (1.0 + np.exp(-(total_return - baseline) / scale))
        boost = 1.0 + lambda_boost * sigmoid_val

        n_boosted = 0
        for pos in positions:
            # Vérifier que la position appartient encore à cet épisode
            if int(self._ep_of_pos[pos]) != episode_id:
                continue
            if pos >= len(self.buffer):
                continue
            self.priorities[pos] = float(
                np.clip(self.priorities[pos] * boost, PER_EPS, MAX_PRIORITY)
            )
            n_boosted += 1

        # Libération mémoire : entrée consommée
        del self._ep_positions[episode_id]

        return n_boosted, boost

    def __len__(self):
        """@brief Retourne le nombre de transitions stockées."""
        return min(self.pos, self.capacity) if len(self.buffer) < self.capacity \
               else self.capacity