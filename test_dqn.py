import time
import torch
import numpy as np
from pathlib import Path

from wrappers import make_train_env
from dqn_model import DQN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = Path(__file__).resolve().parent / "checkpoints" / "mspacman_dqn.pth"

def play():
    # render_mode="human" pour voir le jeu, clip_rewards=False pour avoir le vrai score Atari
    env = make_train_env(render_mode="human", clip_rewards=False)
    
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    model = DQN(obs_shape, n_actions).to(DEVICE)
    
    # Chargement des poids de l'entraînement
    if CKPT_PATH.exists():
        checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["policy_net"])
        print(f"✅ Modèle chargé depuis {CKPT_PATH}")
    else:
        print("⚠️ Aucun fichier d'entraînement trouvé. L'agent va jouer avec des poids aléatoires.")

    model.eval()
    state, _ = env.reset()
    
    total_reward = 0.0
    dots_manges = 0
    ghosts_eaten = 0
    done = False

    print("Début de la partie (Epsilon = 0.0)...")
    
    while not done:
        # Sélection de l'action de manière 100% déterministe (Epsilon = 0)
        with torch.no_grad():
            state_t = torch.from_numpy(np.array(state)).unsqueeze(0).to(DEVICE)
            q_values = model(state_t)
            action = int(q_values.argmax(dim=1).item())

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # --- NOUVEAU : Incrémentation des dots et des fantômes ---
        if reward > 0:
            dots_manges += 1
        if reward >= 200:
            ghosts_eaten += 1

        state = next_state
        total_reward += reward
        
        # Ralentir un peu le rendu pour que ce soit visible à l'oeil nu
        time.sleep(0.02)

    print("\n--- RÉSULTATS DE LA PARTIE ---")
    print(f"Score final    : {total_reward}")
    print(f"Points mangés  : {dots_manges}")
    print(f"Fantômes mangés: {ghosts_eaten}")
    
    if dots_manges >= 148:
        print("\033[92m[FINISH] L'agent a terminé un tableau !\033[0m")
        
    env.close()

if __name__ == "__main__":
    play()
