"""
test.py — Évaluation d'un agent DQN entraîné sur MsPacman-ALE.
Lance une partie complète en mode greedy (ε-greedy configurable)
et affiche les statistiques de la partie.

Usage :
    make test
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from wrappers import make_test_env
from dqn_model import DQN

# ── Constantes ────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
CKPT_PATH  = BASE_DIR / "checkpoints" / "mspacman_dqn.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOTS_LEVEL = 158      # gommes nécessaires pour valider un tableau
MAX_STEPS  = 27_000   # garde-fou anti-boucle infinie


# ── Argument parser ───────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test d'un agent DQN MsPacman",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, default=str(CKPT_PATH),
        help="Chemin vers le fichier .pth",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Affiche la fenêtre de jeu en temps réel",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0,
        help="Epsilon pour la politique ε-greedy lors du test",
    )
    return parser.parse_args()


# ── Chargement robuste des poids ──────────────────────────────────────────────
def load_weights(policy_net: DQN, checkpoint: dict) -> None:
    """Charge les poids depuis le checkpoint.
    Supporte le nouveau format (conv/fc) et l'ancien (net./head.).
    """
    raw_sd = checkpoint.get("policy_net", checkpoint.get("net", checkpoint))

    if not (isinstance(raw_sd, dict) and raw_sd and
            isinstance(next(iter(raw_sd.values())), torch.Tensor)):
        print("[AVERTISSEMENT] Format de checkpoint non reconnu — poids aléatoires.")
        return

    try:
        policy_net.load_state_dict(raw_sd)
    except RuntimeError:
        new_sd = {}
        for k, v in raw_sd.items():
            nk = (k.replace("net.", "conv.")
                   .replace("head.", "fc.")
                   .replace("head.0", "fc.0")
                   .replace("head.2", "fc.2"))
            new_sd[nk] = v
        policy_net.load_state_dict(new_sd, strict=False)
        print("[INFO] Poids convertis depuis l'ancien format (net./head.).")


# ── Boucle de jeu ─────────────────────────────────────────────────────────────
def run_episode(policy_net: DQN, env, epsilon: float) -> tuple[float, int, int, int]:
    """Joue une partie complète.
    Returns: total_reward, step, dots_manges, ghosts_eaten
    """
    state, _ = env.reset()
    total_reward = 0.0
    step         = 0
    dots_manges  = 0
    ghosts_eaten = 0
    done         = False

    while not done and step < MAX_STEPS:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_t = (
                    torch.FloatTensor(state)
                    .unsqueeze(0)
                    .to(DEVICE) / 255.0
                )
                action = policy_net(state_t).argmax(1).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        raw_reward = float(info.get("raw_reward", reward))

        if raw_reward > 0:
            dots_manges += 1
        if raw_reward >= 200:
            ghosts_eaten += 1

        total_reward += raw_reward
        step         += 1
        state         = next_state

    return total_reward, step, dots_manges, ghosts_eaten


# ── Point d'entrée ────────────────────────────────────────────────────────────
def main() -> None:
    args      = parse_args()
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        print(f"[ERREUR] Checkpoint introuvable : {ckpt_path}")
        return

    print(f"\nChargement du checkpoint : {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    config     = checkpoint.get("config", {})
    dueling    = config.get("dueling", False)

    print("\n--- CHECKPOINT ---")
    print(
        f"Config checkpoint : dueling={dueling} "
        f"| double={config.get('double_dqn', '?')} "
        f"| per={config.get('per', '?')} "
        f"| trophy={config.get('trophy_buffer', '?')}"
    )
    print(f"Meilleur score entraînement : {checkpoint.get('best_score', 0):.0f}")
    print(f"Épisodes entraînés          : {checkpoint.get('episode', '?')}")
    print(f"Trophy baseline             : {checkpoint.get('trophy_baseline', 'N/A')}")

    render_mode = "human"
    env         = make_test_env(render_mode=render_mode)
    obs_shape   = env.observation_space.shape
    n_actions   = env.action_space.n

    policy_net = DQN(obs_shape, n_actions, dueling=dueling).to(DEVICE)
    load_weights(policy_net, checkpoint)
    policy_net.eval()

    print(f"\nDispositif : {DEVICE}  |  Architecture : {'Dueling DQN' if dueling else 'DQN classique'}")
    print(f"ε test     : {args.epsilon}  |  Steps max : {MAX_STEPS:,}")

    print("\nDébut de la partie…")
    total_reward, step, dots_manges, ghosts_eaten = run_episode(
        policy_net, env, args.epsilon
    )

    print("\n--- RÉSULTATS DE LA PARTIE ---")
    print(f"Score final     : {total_reward:.0f}")
    print(f"Steps joués     : {step}")
    print(f"Dots mangés     : {dots_manges}")
    print(f"Fantômes mangés : {ghosts_eaten}")

    if dots_manges >= DOTS_LEVEL:
        print("\033[92m★ [FINISH] L'agent a terminé un tableau !\033[0m")
    elif ghosts_eaten > 0:
        print(f"\033[96m{ghosts_eaten} fantôme(s) mangé(s) !\033[0m")

    env.close()


if __name__ == "__main__":
    main()