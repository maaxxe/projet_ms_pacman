"""
@file      plot_progress.py
@brief     Visualise les métriques d'entraînement depuis log.json.
@details   Lit log.json et trace 8 graphiques : score, moyenne 10 épisodes,
           epsilon, loss, steps/épisode, taille buffer, dots mangés, fantômes.
           Lance directement avec : python plot_progress.py
"""

import json
import math
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "log.json"

with open(LOG_PATH, "r", encoding="utf-8") as f:
    log = json.load(f)

if not log:
    raise ValueError("log.json est vide.")

# Tri correct des épisodes : episode_1, episode_2, ...
episode_keys = sorted(log.keys(), key=lambda k: int(k.split("_")[1]))
data = [log[k] for k in episode_keys]

# Extraction des métriques (math.nan si clé absente pour compatibilité)
episodes      = [int(k.split("_")[1]) for k in episode_keys]
epsilons      = [d.get("eps", math.nan) for d in data]
losses        = [d["loss"] if d.get("loss") is not None else math.nan for d in data]
scores        = [d.get("score", d.get("episode_reward", math.nan)) for d in data]
episode_steps = [d.get("steps", d.get("episode_length", math.nan)) for d in data]
avg_score_10  = [d.get("avg10", d.get("avg_reward_10", math.nan)) for d in data]
buffer_sizes  = [d.get("buffer", math.nan) for d in data]
dots_manges   = [d.get("dots", math.nan) for d in data]
ghosts_eaten  = [d.get("ghosts", math.nan) for d in data]
times         = [d.get("timestamp", math.nan) for d in data]

fig, axes = plt.subplots(4, 2, figsize=(15, 16))
axes = axes.ravel()

axes[0].plot(episodes, scores, color="blue")
axes[0].set_title("Score par épisode")
axes[0].set_xlabel("")

axes[1].plot(episodes, avg_score_10, color="orange")
axes[1].set_title("Score moyen (10 derniers)")
axes[1].set_xlabel("")

axes[2].plot(episodes, epsilons, color="green")
axes[2].set_title("Epsilon")
axes[2].set_xlabel("")

axes[3].plot(episodes, losses, color="red")
axes[3].set_title("Loss")
axes[3].set_xlabel("")

axes[4].plot(episodes, episode_steps, color="purple")
axes[4].set_title("Steps par épisode")
axes[4].set_xlabel("")

axes[5].plot(episodes, buffer_sizes, color="brown")
axes[5].set_title("Taille du replay buffer")
axes[5].set_xlabel("")

axes[6].plot(episodes, dots_manges, color="teal")
axes[6].set_title("Dots mangés")
axes[6].set_xlabel("")

axes[7].plot(episodes, ghosts_eaten, color="black")
axes[7].set_title("Fantômes mangés")
axes[7].set_xlabel("")

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
