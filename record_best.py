import time
import torch
import numpy as np
import imageio
from pathlib import Path

from wrappers import make_train_env
from dqn_model import DQN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = Path(__file__).resolve().parent / "checkpoints" / "mspacman_dqn.pth"
VIDEO_DIR = Path(__file__).resolve().parent / "videos"

# Création du dossier pour les vidéos s'il n'existe pas
VIDEO_DIR.mkdir(exist_ok=True)

def record_best_runs(num_episodes, min_dots):
    # L'astuce est ici : render_mode="rgb_array" capture les images sans ouvrir de fenêtre
    env = make_train_env(render_mode="rgb_array", clip_rewards=False)
    
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    model = DQN(obs_shape, n_actions).to(DEVICE)
    
    if CKPT_PATH.exists():
        checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["policy_net"])
        print(f"✅ Modèle chargé !")
        print(f"min_dots recherche = {min_dots}")
    else:
        print("⚠️ Aucun modèle trouvé.")
        return

    model.eval()

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        dots_manges = 0
        frames = [] # On va stocker toutes les images de la partie ici

        print(f"Lancement de la partie {ep}/{num_episodes}...")

        while not done:
            # 1. On capture l'écran actuel et on l'ajoute à la vidéo
            frames.append(env.render())

            # 2. L'IA joue (Epsilon = 0)
            with torch.no_grad():
                state_t = torch.from_numpy(np.array(state)).unsqueeze(0).to(DEVICE)
                q_values = model(state_t)
                action = int(q_values.argmax(dim=1).item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if reward > 0:
                dots_manges += 1

            state = next_state

        print(f"Partie {ep} terminée : {dots_manges} dots mangés.")

        # 3. On ne sauvegarde QUE si la condition est remplie
        if dots_manges >= min_dots:
            video_name = f"pacman_win_{dots_manges}dots_ep{ep}.mp4"
            video_path = VIDEO_DIR / video_name
            print(f"🌟 VICTOIRE ! Enregistrement de {video_name} (Patientez...)")
            
            # FPS=30 ou 60 selon la vitesse à laquelle tu veux regarder la vidéo
            imageio.mimsave(str(video_path), frames, fps=60)
            print("✅ Vidéo sauvegardée !")
        else:
            print("❌ Score insuffisant, vidéo effacée de la mémoire.")

    env.close()
    print("Terminé ! Vérifiez le dossier 'videos'.")

if __name__ == "__main__":
    record_best_runs(num_episodes=50, min_dots=155)
