"""
@file    train.py
@brief   DQN MsPacman COMPLET avec 3 améliorations + CHARGEMENT CHECKPOINT ROBUSTE
         - Double DQN (anti sur-estimation)
         - Dueling DQN (V(s) + A(s,a))
         - Prioritized Replay (sample TD-error)

@version 2.1 - Compatible ancien/nouveau checkpoint
@author  Perplexity AI
@date    2026-03-23
"""

import json
import time
import signal
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from wrappers import make_train_env
from dqn_model import DQN  
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

# CUDA reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False

# =============================================================================
# PATHS & FLAGS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "log.json"
CKPT_DIR = BASE_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
CKPT_PATH = CKPT_DIR / "mspacman_dqn.pth"

SAVE_EVERY_EPISODES = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# HYPERPARAMS
# =============================================================================
TOTAL_EPISODES = 3000*100
BUFFER_SIZE = 100_000
BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-4
LEARNING_STARTS = 20_000
TRAIN_FREQ = 4
TARGET_UPDATE_EVERY = 2_000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = 1500
MAX_EPISODE_STEPS = 20_000
DOTS_LEVEL = 158

# =============================================================================
# FLAGS AMÉLIORATIONS (True=activé)
# =============================================================================
USE_DOUBLE_DQN = True           # Double DQN
USE_DUELING_DQN = True          # Dueling DQN  
USE_PER = True                  # Prioritized Replay

PER_ALPHA = 0.6                 # Priorité exponent
PER_BETA_START = 0.4            # IS weight
PER_EPS = 1e-6                  # Anti zero-prio

running = True

def handle_interrupt(sig, frame):
    """Gestion Ctrl+C"""
    global running
    running = False
signal.signal(signal.SIGINT, handle_interrupt)

def episode_epsilon(episode_idx):
    """Epsilon linéaire décroissant"""
    if episode_idx >= EPS_DECAY_EPISODES: return EPS_END
    return EPS_START + (episode_idx/EPS_DECAY_EPISODES)*(EPS_END-EPS_START)

def ensure_log_file():
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", encoding="utf-8") as f: json.dump({}, f, indent=2)

def load_log():
    ensure_log_file()
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f: 
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except: return {}

def save_log(log_data):
    with open(LOG_PATH, "w", encoding="utf-8") as f: json.dump(log_data, f, indent=2)

def save_checkpoint(policy_net, target_net, optimizer, episode, best_score, 
                   total_env_steps, log_data):
    """Sauvegarde complète"""
    torch.save({
        "policy_net": policy_net.state_dict(),
        "target_net": target_net.state_dict(), 
        "optimizer": optimizer.state_dict(),
        "episode": episode, "best_score": best_score,
        "total_env_steps": total_env_steps, "log_data": log_data,
        "config": {"double_dqn": USE_DOUBLE_DQN, "dueling": USE_DUELING_DQN, "per": USE_PER}
    }, CKPT_PATH)

def load_checkpoint_safely(policy_net, target_net, optimizer, checkpoint):
    """
    Chargement robuste ancien/nouveau format DQN
    Mapping: net.* → conv.*, head.* → fc.*
    """
    try:
        # Nouveau format
        policy_net.load_state_dict(checkpoint["policy_net"])
        target_net.load_state_dict(checkpoint["target_net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        return True, "nouveau"
    except RuntimeError:
        try:
            # Ancien format → mapping
            old_dict = checkpoint.get("policy_net", checkpoint.get("net", checkpoint))
            new_dict = {}
            for k, v in old_dict.items():
                new_k = k.replace("net.", "conv.").replace("head.", "fc.")
                new_k = new_k.replace("head.0", "fc.0").replace("head.2", "fc.2")
                new_dict[new_k] = v
            
            policy_net.load_state_dict(new_dict, strict=False)
            target_net.load_state_dict(new_dict, strict=False)
            return True, "ancien_convert"
        except:
            print("❌ Reset: checkpoint incompatible")
            return False, "reset"

def train():
    """Entraînement principal"""
    log_data = load_log()
    env = make_train_env(render_mode=None, clip_rewards=True)
    
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    
    # Réseaux avec flags
    print(f"  Config: DoubleDQN={USE_DOUBLE_DQN}, Dueling={USE_DUELING_DQN}, PER={USE_PER}")
    policy_net = DQN(obs_shape, n_actions, dueling=USE_DUELING_DQN).to(DEVICE)
    target_net = DQN(obs_shape, n_actions, dueling=USE_DUELING_DQN).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    
    # Buffer
    if USE_PER:
        replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=PER_ALPHA)
        print(" PER activé")
    else:
        replay_buffer = ReplayBuffer(BUFFER_SIZE)
    
    # Chargement intelligent
    episode = len(log_data)
    total_env_steps = 0
    best_score = max([v.get("score", -float("inf")) for v in log_data.values()], default=-float("inf"))
    
    if CKPT_PATH.exists():
        checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
        success, mode = load_checkpoint_safely(policy_net, target_net, optimizer, checkpoint)
        if success:
            total_env_steps = checkpoint.get("total_env_steps", 0)
            best_score = max(best_score, checkpoint.get("best_score", -float("inf")))
            episode = checkpoint.get("episode", episode)
            print(f" Checkpoint {mode} chargé | ép={episode} | steps={total_env_steps:,} | best={best_score:.0f}")
        else:
            print(" Reset complet")
    
    current_learning_starts = BATCH_SIZE if total_env_steps > 0 else LEARNING_STARTS
    recent_scores = []
    losses_list = []
    start_time = time.time()
    
    print(f" Début | eps={episode_epsilon(episode+1):.3f} | device={DEVICE}")
    
    # BOUCLE ÉPISODES
    for episode_idx in range(episode, TOTAL_EPISODES):
        if not running: break
        
        state, _ = env.reset()
        ep_reward, ep_score, ep_steps = 0.0, 0.0, 0
        dots_manges, ghosts_eaten = 0, 0
        
        epsilon = episode_epsilon(episode_idx)
        done = False
        
        while not done and ep_steps < MAX_EPISODE_STEPS:
            # Action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE) / 255.0
                    action = policy_net(state_t).argmax(1).item()
            
            next_state, reward, term, trunc, info = env.step(action)
            done = term or trunc
            
            raw_reward = float(info.get("raw_reward", reward))
            if raw_reward > 0: dots_manges += 1
            if raw_reward >= 200: ghosts_eaten += 1
            
            replay_buffer.push(state, action, reward, next_state, done)
            
            state, ep_reward, ep_score, ep_steps = next_state, ep_reward+reward, ep_score+raw_reward, ep_steps+1
            total_env_steps += 1
        
        # APPRENTISSAGE
        if len(replay_buffer) >= current_learning_starts and total_env_steps % TRAIN_FREQ == 0:
            if USE_PER:
                states, actions, rewards, nexts, dones, weights, idxs = replay_buffer.sample(BATCH_SIZE, PER_BETA_START)
            else:
                states, actions, rewards, nexts, dones = replay_buffer.sample(BATCH_SIZE)
                weights, idxs = np.ones(BATCH_SIZE), None
            
            states_t = torch.FloatTensor(states).to(DEVICE) / 255.0
            acts_t = torch.LongTensor(actions).to(DEVICE).unsqueeze(1)
            rews_t = torch.FloatTensor(rewards).to(DEVICE)
            nexts_t = torch.FloatTensor(nexts).to(DEVICE) / 255.0
            dones_t = torch.FloatTensor(dones).to(DEVICE)
            w_t = torch.FloatTensor(weights).to(DEVICE)
            
            q_curr = policy_net(states_t).gather(1, acts_t).squeeze(1)
            
            with torch.no_grad():
                if USE_DOUBLE_DQN:
                    next_act = policy_net(nexts_t).argmax(1, keepdim=True)
                    next_q = target_net(nexts_t).gather(1, next_act).squeeze(1)
                else:
                    next_q = target_net(nexts_t).max(1)[0]
                target_q = rews_t + GAMMA * next_q * (1 - dones_t)
            
            td_err = nn.SmoothL1Loss(reduction='none')(q_curr, target_q)
            loss = (w_t * td_err).mean()
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
            optimizer.step()
            
            if USE_PER:
                replay_buffer.update_priorities(idxs, td_err.detach().cpu().numpy())
            
            losses_list.append(loss.item())
        
        # TARGET UPDATE
        if total_env_steps % TARGET_UPDATE_EVERY == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # LOGS
        recent_scores.append(ep_score)
        if len(recent_scores) > 10: recent_scores.pop(0)
        avg_score = np.mean(recent_scores)
        mean_loss = np.mean(losses_list[-100:]) if losses_list else 0
        
        log_data[f"ep_{episode_idx}"] = {
            "eps": epsilon, "loss": mean_loss, "reward": ep_reward, "score": ep_score,
            "dots": dots_manges, "ghosts": ghosts_eaten, "steps": ep_steps,
            "avg10": avg_score, "buffer": len(replay_buffer)
        }
        
        if dots_manges >= DOTS_LEVEL:
            print(f"\033[92m🎉 LEVEL CLEAR! ({dots_manges}/{DOTS_LEVEL})\033[0m")
        
        print(f"Ep {episode_idx:4d} | score={ep_score:6.1f} | dots={dots_manges:3d} "
              f"| ghosts={ghosts_eaten} | eps={epsilon:.3f} | loss={mean_loss:.3f}")
        
        if ep_score > best_score:
            best_score = ep_score
            print(f" NEW BEST: {best_score:.0f}")
        
        if episode_idx % SAVE_EVERY_EPISODES == 0:
            save_log(log_data)
            save_checkpoint(policy_net, target_net, optimizer, episode_idx, 
                          best_score, total_env_steps, log_data)
            print(f" Saved ep{episode_idx}")
    
    save_checkpoint(policy_net, target_net, optimizer, episode_idx, best_score, 
                   total_env_steps, log_data)
    env.close()
    print(" Training terminé")

if __name__ == "__main__":
    train()
