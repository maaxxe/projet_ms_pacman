"""
@file dqn_model.py
@brief Réseau DQN avec support Dueling Architecture (optionnel)
"""

import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    @class DQN
    @brief Réseau DQN MsPacman (CNN + FC)
    Supporte Dueling DQN si USE_DUELING_DQN=True
    """
    def __init__(self, input_shape, n_actions, dueling=False):
        """
        @param input_shape (C,H,W) shape observation (4,84,84)
        @param n_actions 9 actions MsPacman
        @param dueling Active Dueling heads (V(s) + A(s,a))
        """
        super(DQN, self).__init__()
        c, h, w = input_shape
        
        # Features CNN (commun aux deux variantes)
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        
        # Calcule flatten dim
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flatten = self.conv(dummy).shape[1]
        
        if dueling:
            # DUELING: deux heads séparés
            self.value_stream = nn.Sequential(  # V(s): valeur état
                nn.Linear(n_flatten, 512), nn.ReLU(),
                nn.Linear(512, 1)
            )
            self.advantage_stream = nn.Sequential(  # A(s,a): avantage action
                nn.Linear(n_flatten, 512), nn.ReLU(),
                nn.Linear(512, n_actions)
            )
        else:
            # CLASSIQUE: Q(s,a) direct
            self.fc = nn.Sequential(
                nn.Linear(n_flatten, 512), nn.ReLU(),
                nn.Linear(512, n_actions)
            )
    
    def forward(self, x):
        """
        @brief Forward pass
        @param x Batch d'observations (B,4,84,84) [0-255 uint8]
        @return Q-values (B, n_actions)
        """
        x = x.float() / 255.0
        features = self.conv(x)
        
        if hasattr(self, 'value_stream'):  # Dueling
            V = self.value_stream(features)  # (B,1)
            A = self.advantage_stream(features)  # (B,9)
            Q = V + (A - A.mean(dim=1, keepdim=True))  # Mean subtraction
            return Q
        else:  # Classique
            return self.fc(features)
