"""
@file      dqn_model.py
@brief     Réseau de neurones DQN convolutif pour MsPacman.
@details   Supporte deux architectures selon le flag USE_DUELING_DQN dans train.py :
           - Classique : CNN → fc → Q(s,a)
           - Dueling    : CNN → [V(s)] + [A(s,a)] → Q(s,a) = V + (A - mean(A))
           L'entrée est un stack de 4 frames 84x84 en niveaux de gris.
"""

import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    @class   DQN
    @brief   Réseau principal utilisé comme policy_net ET target_net.
    @details Si dueling=True (USE_DUELING_DQN=True dans train.py), le réseau
             sépare l'estimation de la valeur d'état V(s) et des avantages par
             action A(s,a) pour former Q(s,a) = V(s) + (A(s,a) - mean(A(s,*))).
             Cela améliore la généralisation dans les états où les actions ont
             des effets similaires (corridors Pacman).
    """

    def __init__(self, input_shape, n_actions, dueling=False):
        """
        @brief   Initialise le réseau CNN + tête(s) fully-connected.
        @param   input_shape  Tuple (C, H, W) = (4, 84, 84) : 4 frames empilées.
        @param   n_actions    Nombre d'actions discrètes = 9 pour MsPacman.
        @param   dueling      Si True, active l'architecture Dueling (USE_DUELING_DQN).
                              Si False, architecture DQN classique (fc unique).
        """
        super(DQN, self).__init__()
        c, h, w = input_shape

        # Backbone CNN commun aux deux variantes (3 couches conv)
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )

        # Calcul dynamique de la dimension aplatie après les convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flatten = self.conv(dummy).shape[1]

        if dueling:
            # DUELING (USE_DUELING_DQN=True) : deux streams séparés
            # value_stream estime V(s) : "à quel point cet état est bon ?"
            self.value_stream = nn.Sequential(
                nn.Linear(n_flatten, 512), nn.ReLU(),
                nn.Linear(512, 1)           # scalaire V(s)
            )
            # advantage_stream estime A(s,a) : "quelle action est relativement meilleure ?"
            self.advantage_stream = nn.Sequential(
                nn.Linear(n_flatten, 512), nn.ReLU(),
                nn.Linear(512, n_actions)   # vecteur A(s,a) pour chaque action
            )
        else:
            # CLASSIQUE (USE_DUELING_DQN=False) : Q(s,a) direct
            self.fc = nn.Sequential(
                nn.Linear(n_flatten, 512), nn.ReLU(),
                nn.Linear(512, n_actions)
            )

    def forward(self, x):
        """
        @brief   Calcule les Q-values pour un batch d'observations.
        @param   x  Tensor (B, 4, 84, 84) en uint8 [0-255], normalisé à [0,1].
        @return  Tensor (B, n_actions) de Q-values.
        @details Si USE_DUELING_DQN=True : Q = V(s) + (A(s,a) - mean_a(A(s,a)))
                 La soustraction de la moyenne centre les avantages et stabilise.
                 Si USE_DUELING_DQN=False : Q = fc(features).
        """
        x = x.float() / 255.0
        features = self.conv(x)

        if hasattr(self, 'value_stream'):  # Dueling actif
            V = self.value_stream(features)                        # (B, 1)
            A = self.advantage_stream(features)                    # (B, n_actions)
            Q = V + (A - A.mean(dim=1, keepdim=True))             # mean subtraction
            return Q
        else:                              # Classique
            return self.fc(features)
