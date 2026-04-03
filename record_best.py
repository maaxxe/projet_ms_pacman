"""
@file      record_best.py
@brief     Enregistre en vidéo MP4 les meilleures parties de l'agent.
@details   Lance N épisodes et sauvegarde uniquement ceux où dots_manges >= DOTS_LEVEL.
           Utilise render_mode="rgb_array" pour capturer sans fenêtre.
           Vidéos sauvegardées dans ./videos/pacman_win_Xdots_epY.mp4
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from wrappers import make_test_env
from dqn_model import DQN

BASE_DIR = Path(__file__).resolve().parent
CKPT_PATH = BASE_DIR / "checkpoints" / "mspacman_dqn.pth"
VIDEO_DIR = BASE_DIR / "videos"
VIDEO_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOTS_LEVEL = 158

NUM_EPISODES = 100
EPSILON      = 0.0
FPS          = 30
MAX_STEPS    = 27_000




def load_model(ckpt_path, obs_shape, n_actions):
    dueling = True
    checkpoint = None

    if Path(ckpt_path).exists():
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        config = checkpoint.get("config", {})
        dueling = config.get("dueling", True)
        print(f"Checkpoint : {Path(ckpt_path).name}")
        print(f"Dueling    : {dueling}")
        print(f"Best score : {checkpoint.get('best_score', 0):.0f}")
        print(f"Trophy BL  : {checkpoint.get('trophy_baseline', 'N/A')}")
    else:
        print(f"[ERREUR] Checkpoint introuvable : {ckpt_path}")
        return None

    model = DQN(obs_shape, n_actions, dueling=dueling).to(DEVICE)
    model.load_state_dict(checkpoint["policy_net"])
    model.eval()
    return model


def select_action(model, state, epsilon):
    if epsilon > 0 and np.random.rand() < epsilon:
        return None

    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE) / 255.0
        return int(model(state_t).argmax(1).item())


def run_episode(env, model, epsilon=0.0, max_steps=MAX_STEPS):
    state, _ = env.reset()
    frames = []
    total_reward = 0.0
    dots_manges = 0
    ghosts_eaten = 0
    steps = 0
    done = False

    first_frame = env.render()
    if first_frame is not None:
        frames.append(np.array(first_frame).copy())

    while not done and steps < max_steps:
        action = select_action(model, state, epsilon)
        if action is None:
            action = env.action_space.sample()

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        frame = env.render()
        if frame is not None:
            frames.append(np.array(frame).copy())

        raw_reward = float(info.get("raw_reward", reward))
        if 0 < raw_reward <= 50:
            dots_manges += 1
        if raw_reward >= 200:
            ghosts_eaten += 1

        total_reward += raw_reward
        state = next_state
        steps += 1

    return {
        "frames": frames,
        "score": total_reward,
        "dots": dots_manges,
        "ghosts": ghosts_eaten,
        "steps": steps,
        "level_clear": dots_manges >= DOTS_LEVEL,
    }


def main():
    MAX_IN_RUN = 0

    env = make_test_env(render_mode="rgb_array")
    model = load_model(CKPT_PATH, env.observation_space.shape, env.action_space.n)
    if model is None:
        env.close()
        return

    saved = 0
    print(f"\nEpisodes : {NUM_EPISODES} | DOTS_LEVEL : {DOTS_LEVEL} | epsilon : {EPSILON}")
    print("-" * 60)

    try:
        for ep in range(1, NUM_EPISODES + 1):
            out = run_episode(env, model, epsilon=EPSILON)
            if (out['dots']>= MAX_IN_RUN ):
                MAX_IN_RUN = out['dots']
            print(
                f"Ep {ep:3d}/{NUM_EPISODES} | score={out['score']:7.0f} | "
                f"dots={out['dots']:3d} | ghosts={out['ghosts']} | steps={out['steps']:5d}"
            )

            if out["dots"] >= DOTS_LEVEL:
                saved += 1
                video_name = f"pacman_win_{out['dots']}dots_ep{ep}.mp4"
                video_path = VIDEO_DIR / video_name
                imageio.mimsave(str(video_path), out["frames"], fps=FPS)
                print(f"  -> saved {video_name}")
            else:
                print("  -> skipped")
    finally:
        env.close()

    print(f"Terminé. Vidéos sauvegardées: {saved}")
    print(f"Max_enregistré : {MAX_IN_RUN}")


if __name__ == "__main__":
    main()