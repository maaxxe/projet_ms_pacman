# wrappers.py
import collections
import json
import numpy as np
import gymnasium as gym
import ale_py
from gymnasium import spaces
from PIL import Image
from pathlib import Path

gym.register_envs(ale_py)

BASE_DIR = Path(__file__).resolve().parent
REWARDS_FILE = BASE_DIR / "reward.json"

def load_level_bonus():
    if REWARDS_FILE.exists():
        with open(REWARDS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {"level_clear_bonus": 50.0, "dots_level": 158} # il en reste

# --- ENVIRONNEMENT QUI RAJOUTE JUSTE LE BONUS DE NIVEAU ---
class LevelClearBonusEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.dots_eaten = 0
        self.config = load_level_bonus()

    def reset(self, **kwargs):
        self.config = load_level_bonus() # Recharge le json à chaque partie
        self.dots_eaten = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # On utilise le score brut (sauvegardé par MaxAndSkipEnv) pour compter
        raw_score = info.get("raw_reward", reward)
        
        # 10 pts = pastille normale | 50 pts = super-pastille
        if raw_score == self.config["pastille_normale"] or raw_score == self.config["pastille_grosse"]:
            self.dots_eaten += 1
            
        # Si on a tout mangé, on AJOUTE le bonus à la récompense actuelle
        if self.dots_eaten == self.config["dots_level"]:
            reward += self.config["level_clear_bonus"]
            print(f"\033[93m[REWARD] Jackpot Level Clear ! (+{self.config['level_clear_bonus']})\033[0m")
            # On remet à zéro pour le niveau 2
            self.dots_eaten = 0 

        return obs, reward, terminated, truncated, info

# --- AUTRES WRAPPERS ---
class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()
        self.has_fire = "FIRE" in meanings

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.has_fire:
            fire_idx = self.env.unwrapped.get_action_meanings().index("FIRE")
            obs, _, terminated, truncated, info = self.env.step(fire_idx)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = collections.deque(maxlen=2)

    def step(self, action):
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
        info["raw_reward"] = float(total_reward)

        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info

class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )

    def observation(self, obs):
        img = Image.fromarray(obs)
        img = img.convert("L")
        img = img.resize((84, 84), Image.Resampling.BILINEAR)
        return np.array(img, dtype=np.uint8)

class FrameStack(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = collections.deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(k, shp[0], shp[1]), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.stack(self.frames, axis=0)

class ClipRewardEnv(gym.RewardWrapper):
    def reward(self, reward):
        return np.sign(reward)

def _make_base_env(render_mode=None):
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
    env = _make_base_env(render_mode=render_mode)
    env = FireResetEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    
    # On glisse juste notre bonus de fin de niveau ici
    env = LevelClearBonusEnv(env)
    
    env = ProcessFrame84(env)
    env = FrameStack(env, k=4)

    if clip_rewards:
        env = ClipRewardEnv(env)

    return env

def make_test_env(render_mode=None):
    env = _make_base_env(render_mode=render_mode)
    env = MaxAndSkipEnv(env, skip=4)
    env = ProcessFrame84(env)
    env = FrameStack(env, k=4)
    return env

def make_env(render_mode=None, clip_rewards=True):
    return make_train_env(render_mode=render_mode, clip_rewards=clip_rewards)
