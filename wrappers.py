"""
@file      wrappers.py
@brief     Wrappers Gymnasium pour MsPacman (ALE/MsPacman-v5).
@details   Chaîne de wrappers appliquée dans make_train_env() :
           FireResetEnv → MaxAndSkipEnv → LevelClearBonusEnv
           → ProcessFrame84 → FrameStack(4) → ClipRewardEnv

           Chaque wrapper transforme obs, reward ou reset indépendamment,
           permettant de les combiner/désactiver facilement.
"""

import collections
import json
import numpy as np
import gymnasium as gym
import ale_py
from gymnasium import spaces
from PIL import Image
from pathlib import Path

gym.register_envs(ale_py)

BASE_DIR      = Path(__file__).resolve().parent
REWARDS_FILE  = BASE_DIR / "reward.json"


def load_level_bonus():
    """
    @brief  Charge la config de bonus niveau depuis reward.json.
    @return Dict avec keys level_clear_bonus, dots_level, pastille_normale, pastille_grosse.
    @details Si reward.json absent, retourne des valeurs par défaut.
    """
    if REWARDS_FILE.exists():
        with open(REWARDS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {"level_clear_bonus": 50.0, "dots_level": 158, "pastille_normale": 10,"pastille_grosse": 50 }


class LevelClearBonusEnv(gym.Wrapper):
    """
    @class   LevelClearBonusEnv
    @brief   Ajoute un bonus de récompense quand le niveau est terminé.
    @details Compte les gommes mangées via raw_reward (10=normale, 50=grosse).
             Quand dots_eaten == config["dots_level"] (158), ajoute level_clear_bonus
             à la récompense et remet dots_eaten à 0 pour le niveau suivant.
    """

    def __init__(self, env):
        """
        @brief  Initialise avec config chargée depuis reward.json.
        @param  env  Environnement Gymnasium à wrapper.
        """
        super().__init__(env)
        self.dots_eaten = 0
        self.config = load_level_bonus()

    def reset(self, **kwargs):
        """
        @brief  Recharge reward.json (hot-reload) et remet dots_eaten à 0.
        @return obs, info du reset.
        """
        self.config = load_level_bonus()
        self.dots_eaten = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        @brief  Step avec détection finish level et injection du bonus.
        @param  action  Action à exécuter.
        @return obs, reward (augmenté si level clear), terminated, truncated, info.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw_score = info.get("raw_reward", reward)

        # 10 pts = pastille normale | 50 pts = super-pastille
        if raw_score == self.config["pastille_normale"] or raw_score == self.config["pastille_grosse"]:
            self.dots_eaten += 1

        # Bonus injecté quand tous les dots du niveau sont mangés
        if self.dots_eaten == self.config["dots_level"]:
            reward += self.config["level_clear_bonus"]
            print(f"\033[93m[REWARD] Jackpot Level Clear ! (+{self.config['level_clear_bonus']})\033[0m")
            self.dots_eaten = 0  # Reset pour le niveau 2

        return obs, reward, terminated, truncated, info


class FireResetEnv(gym.Wrapper):
    """
    @class   FireResetEnv
    @brief   Exécute l'action FIRE au reset pour démarrer la partie.
    @details Certains jeux Atari (dont MsPacman) nécessitent FIRE pour lancer
             la balle ou démarrer. Sans ça, l'agent reste immobile au début.
    """

    def __init__(self, env):
        """@param env Environnement à wrapper."""
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()
        self.has_fire = "FIRE" in meanings

    def reset(self, **kwargs):
        """
        @brief  Reset + action FIRE si disponible.
        @return obs, info après FIRE.
        """
        obs, info = self.env.reset(**kwargs)
        if self.has_fire:
            fire_idx = self.env.unwrapped.get_action_meanings().index("FIRE")
            obs, _, terminated, truncated, info = self.env.step(fire_idx)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """
    @class   MaxAndSkipEnv
    @brief   Répète l'action N fois et retourne le max des 2 dernières frames.
    @details Technique standard Atari : accélère l'entraînement (skip=4 actions
             par step agent) et élimine le flickering des sprites avec max pooling.
             Sauvegarde raw_reward (somme des récompenses sur le skip) dans info.
    """

    def __init__(self, env, skip=4):
        """
        @param  env   Environnement à wrapper.
        @param  skip  Nombre de frames répétées par action (défaut 4).
        """
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = collections.deque(maxlen=2)

    def step(self, action):
        """
        @brief  Répète action skip fois, accumule reward, max des 2 dernières obs.
        @param  action  Action à répéter.
        @return max_frame, total_reward, terminated, truncated, info (+ raw_reward).
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break

        if len(self._obs_buffer) == 2:
            max_frame = np.maximum(self._obs_buffer[0], self._obs_buffer[1])
        else:
            max_frame = self._obs_buffer[-1]

        info = dict(info)
        info["raw_reward"] = float(total_reward)  # Score brut (non-clippé)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        """@brief Reset + vide le buffer de frames."""
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info


class ProcessFrame84(gym.ObservationWrapper):
    """
    @class   ProcessFrame84
    @brief   Convertit l'observation RGB en grayscale 84x84.
    @details Réduit la dimensionnalité : 210x160x3 (RGB) → 84x84x1 (gray).
             Utilisé avant FrameStack pour empiler 4 frames grises.
    """

    def __init__(self, env):
        """@param env Environnement à wrapper."""
        super().__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)

    def observation(self, obs):
        """
        @brief  Transforme obs RGB → grayscale 84x84.
        @param  obs  Frame RGB brute.
        @return np.array uint8 (84, 84).
        """
        img = Image.fromarray(obs)
        img = img.convert("L")
        img = img.resize((84, 84), Image.Resampling.BILINEAR)
        return np.array(img, dtype=np.uint8)


class FrameStack(gym.Wrapper):
    """
    @class   FrameStack
    @brief   Empile les k dernières frames en un seul tenseur (k, H, W).
    @details Donne au réseau le sens du mouvement (fantômes, Pacman).
             Par défaut k=4 → observation shape = (4, 84, 84).
    """

    def __init__(self, env, k=4):
        """
        @param  env  Environnement à wrapper.
        @param  k    Nombre de frames à empiler (défaut 4).
        """
        super().__init__(env)
        self.k = k
        self.frames = collections.deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(k, shp[0], shp[1]), dtype=np.uint8
        )

    def reset(self, **kwargs):
        """@brief Reset + remplit le buffer avec k copies de la première obs."""
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        """@brief Step + ajoute la nouvelle obs au buffer."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        """@brief Empile les k frames en (k, H, W)."""
        return np.stack(self.frames, axis=0)


class ClipRewardEnv(gym.RewardWrapper):
    """
    @class   ClipRewardEnv
    @brief   Clippe la récompense à {-1, 0, +1} via np.sign().
    @details Technique standard Atari : normalise les récompenses entre jeux.
             Activé si clip_rewards=True dans make_train_env().
             Désactivé pour le test (make_test_env) pour avoir le vrai score.
    """
    def reward(self, reward):
        """@brief Retourne np.sign(reward) : -1, 0 ou +1."""
        return np.sign(reward)


def _make_base_env(render_mode=None):
    """
    @brief  Crée l'environnement ALE MsPacman de base.
    @param  render_mode  None (entraînement) ou "human"/"rgb_array" (test/record).
    @return env Gymnasium brut avec frameskip=1.
    """
    env = gym.make(
        "ALE/MsPacman-v5",
        render_mode=render_mode,
        frameskip=1,
        full_action_space=False
    )
    try:
        env.unwrapped.ale.setBool("sound", False)
        env.unwrapped.ale.setBool("display_screen", False)
    except Exception:
        pass
    return env


def make_train_env(render_mode=None, clip_rewards=True):
    """
    @brief  Construit l'environnement complet pour l'entraînement.
    @param  render_mode   Mode rendu (None = headless).
    @param  clip_rewards  Si True, clippe rewards à {-1,0,+1} .
    @return env Gymnasium prêt pour DQN.
    """
    env = _make_base_env(render_mode=render_mode)
    env = FireResetEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    if clip_rewards:
        env = ClipRewardEnv(env)       
    env = LevelClearBonusEnv(env) 
    env = ProcessFrame84(env)
    env = FrameStack(env, k=4)
   
    return env


def make_test_env(render_mode=None):
    """
    @brief  Environnement de test (sans ClipReward, sans FireReset, sans bonus).
    @param  render_mode  "human" pour visualiser, "rgb_array" pour record.
    @return env Gymnasium pour évaluation.
    """
    env = _make_base_env(render_mode=render_mode)
    env = MaxAndSkipEnv(env, skip=4)
    env = ProcessFrame84(env)
    env = FrameStack(env, k=4)
    return env


def make_env(render_mode=None, clip_rewards=True):
    """@brief Alias de make_train_env() pour compatibilité."""
    return make_train_env(render_mode=render_mode, clip_rewards=clip_rewards)
