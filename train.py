# train.py
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
from replay_buffer import ReplayBuffer

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "log.json"
CKPT_DIR = BASE_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
CKPT_PATH = CKPT_DIR / "mspacman_dqn.pth"

SAVE_EVERY_EPISODES = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOTAL_EPISODES = 3000
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

running = True

def handle_interrupt(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, handle_interrupt)

def episode_epsilon(episode_idx):
    if episode_idx >= EPS_DECAY_EPISODES:
        return EPS_END
    frac = episode_idx / EPS_DECAY_EPISODES
    return EPS_START + frac * (EPS_END - EPS_START)

def ensure_log_file():
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)

def load_log():
    ensure_log_file()
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def save_log(log_data):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

def save_checkpoint(policy_net, target_net, optimizer, episode, best_score, total_env_steps, log_data):
    torch.save({
        "policy_net": policy_net.state_dict(),
        "target_net": target_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode": episode,
        "best_score": best_score,
        "total_env_steps": total_env_steps,
        "log_data": log_data,
    }, CKPT_PATH)

def train():
    log_data = load_log()
    env = make_train_env(render_mode=None, clip_rewards=True)

    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    policy_net = DQN(obs_shape, n_actions).to(DEVICE)
    target_net = DQN(obs_shape, n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    
    # --- INITIALISATION ET CHARGEMENT ---
    episode = len(log_data)
    total_env_steps = 0
    best_score = max(
        [v.get("score", v.get("episode_reward", -float("inf"))) for v in log_data.values()],
        default=-float("inf")
    )

    if CKPT_PATH.exists():
        checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
        policy_net.load_state_dict(checkpoint["policy_net"])
        target_net.load_state_dict(checkpoint["target_net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        
        # Récupération des variables cruciales pour reprendre l'entraînement
        total_env_steps = checkpoint.get("total_env_steps", 0)
        best_score = max(best_score, checkpoint.get("best_score", -float("inf")))
        
        print(f" Poids rechargés avec succès ! (Épisode: {episode} | Steps globaux: {total_env_steps})")
    else:
        print(" Aucun fichier de sauvegarde trouvé, démarrage de zéro.")

    # Si l'IA a déjà un historique, on bypass les 20 000 steps d'échauffement 
    # pour pouvoir recommencer à apprendre dès que le buffer a 32 éléments.
    current_learning_starts = LEARNING_STARTS
    if total_env_steps > 0:
        current_learning_starts = BATCH_SIZE
        print("ℹ L'agent étant déjà entraîné, l'apprentissage reprendra immédiatement. "+"\n")

    if episode > 0:
        print(f"Reprise entraînement | episode={episode}")
    else:
        print("Démarrage d'un nouvel entraînement.")

    recent_scores = []
    losses = []
    start_time = time.time()

    # --- BOUCLE PRINCIPALE ---
    for _ in range(TOTAL_EPISODES):
        if not running:
            break

        state, _ = env.reset()
        episode_reward = 0.0
        episode_score = 0.0
        episode_steps = 0
        dots_manges = 0
        ghosts_eaten = 0

        current_episode = episode + 1
        epsilon = episode_epsilon(current_episode)

        done = False
        while not done and episode_steps < MAX_EPISODE_STEPS and running:
            # Choix de l'action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.from_numpy(np.array(state)).unsqueeze(0).to(DEVICE)
                    q_values = policy_net(state_t)
                    action = int(q_values.argmax(dim=1).item())

            # Interaction avec l'environnement
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            raw_reward = float(info.get("raw_reward", reward))

            if raw_reward > 0:
                dots_manges += 1

            if raw_reward >= 200:
                ghosts_eaten += 1

            # Sauvegarde dans le buffer
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            episode_score += raw_reward
            episode_steps += 1
            total_env_steps += 1

            # --- APPRENTISSAGE ---
            if len(replay_buffer) >= current_learning_starts and total_env_steps % TRAIN_FREQ == 0:
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

                states = torch.from_numpy(states).to(DEVICE)
                actions = torch.from_numpy(actions).to(DEVICE).unsqueeze(1)
                rewards = torch.from_numpy(rewards).to(DEVICE)
                next_states = torch.from_numpy(next_states).to(DEVICE)
                dones = torch.from_numpy(dones).to(DEVICE)

                q_values = policy_net(states).gather(1, actions).squeeze(1)

                with torch.no_grad():
                    next_q_values = target_net(next_states).max(dim=1)[0]
                    targets = rewards + GAMMA * next_q_values * (1.0 - dones)

                loss = nn.SmoothL1Loss()(q_values, targets)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()

                losses.append(float(loss.item()))

            # --- MISE A JOUR DU RESEAU CIBLE ---
            if total_env_steps % TARGET_UPDATE_EVERY == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # --- FIN D'EPISODE ---
        recent_scores.append(episode_score)
        if len(recent_scores) > 10:
            recent_scores.pop(0)

        avg_score_10 = float(np.mean(recent_scores))
        mean_loss = float(np.mean(losses[-100:])) if losses else None

        episode += 1

        log_data[f"episode_{episode}"] = {
            "epsilon": float(epsilon),
            "loss": None if mean_loss is None else float(mean_loss),
            "reward_clipped": float(episode_reward),
            "score": float(episode_score),
            "dots_manges": int(dots_manges),
            "ghosts_eaten": int(ghosts_eaten),
            "episode_steps": int(episode_steps),
            "avg_score_10": float(avg_score_10),
            "buffer_size": int(len(replay_buffer)),
            "timestamp": float(time.time() - start_time)
        }
        
        if (dots_manges >= 148):
            print("\033[92m[FINISH] L'agent a terminé un tableau !\033[0m")
            
        print(
            f"Episode {episode} | step={episode_steps} "
            f"| score={episode_score:.1f} | ghosts={ghosts_eaten} | dots={dots_manges} "
            f"| eps={epsilon:.4f} "
            f"| buffer={len(replay_buffer)} | loss={mean_loss if mean_loss is not None else 'None'}"
        )

        if episode_score > best_score:
            best_score = episode_score

        if episode % SAVE_EVERY_EPISODES == 0:
            save_log(log_data)
            save_checkpoint(
                policy_net, target_net, optimizer,
                episode, best_score, total_env_steps, log_data
            )
            print(f"[save] episode={episode} | file={CKPT_PATH.name}")

    # --- FIN DE L'ENTRAÎNEMENT (INTERRUPTION) ---
    save_log(log_data)
    save_checkpoint(
        policy_net, target_net, optimizer,
        episode, best_score, total_env_steps, log_data
    )

    env.close()
    print(f"Training stopped and state saved in {CKPT_PATH}")

if __name__ == "__main__":
    train()
