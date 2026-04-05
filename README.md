# DQN MsPacman — Entraînement par renforcement profond

> Agent DQN entraîné sur `ALE/MsPacman-v5` avec quatre améliorations cumulables,
> dont deux contributions originales : le **Trophy Buffer** et un **système de récompenses configurable**.

---

## Présentation du projet

Ce projet implémente un agent Deep Q-Network (DQN) capable de jouer à Ms. Pac-Man via la librairie ALE (Arcade Learning Environment). L'objectif est de finir un niveau en mangeant toutes les gommes, en combinant plusieurs techniques d'apprentissage par renforcement profond, avec deux innovations maison qui améliorent sensiblement la qualité d'apprentissage.

Le projet est entièrement écrit en Python/PyTorch et supporte la reprise automatique depuis un checkpoint.

---

## Architecture du projet

```
mspacman-dqn/
├── train.py              ← Boucle d'entraînement principale (flags ON/OFF)
├── dqn_model.py          ← Réseau CNN (classique ou Dueling)
├── replay_buffer.py      ← ReplayBuffer uniforme + PrioritizedReplayBuffer (PER)
├── wrappers.py           ← Chaîne de wrappers Gymnasium
├── reward.json           ← Récompenses personnalisées (hot-reload)
├── test_dqn.py           ← Évaluation de l'agent
├── record_best.py        ← Enregistrement vidéo des meilleurs épisodes
├── plot_progress.py      ← Visualisation des courbes d'apprentissage
├── makefile              ← Commandes de lancement
├── checkpoints/          ← Sauvegardes du modèle (.pth)
├── videos/               ← Enregistrements des meilleures parties
└── log.json              ← Historique des épisodes (score, dots, ghosts…)
```

---

## Pipeline de traitement de l'environnement

L'observation brute `210×160 RGB` de l'émulateur passe par une chaîne de wrappers avant d'atteindre le réseau :

```
ALE/MsPacman-v5 (frameskip=1)
      ↓
FireResetEnv         — envoie FIRE au reset pour démarrer la partie
      ↓
MaxAndSkipEnv        — répète l'action ×4, max-pool des 2 dernières frames
      ↓                   applique les récompenses custom depuis reward.json
LifeLostPenaltyEnv   — pénalité –1 à chaque vie perdue
      ↓
LevelClearBonusEnv   — bonus +50 quand les 158 gommes d'un niveau sont mangées  
      ↓
ProcessFrame84       — conversion RGB → grayscale 84×84
      ↓
FrameStack(k=4)      — empilement de 4 frames → tenseur (4, 84, 84)
```

> Les wrappers `test_env` (utilisés pour l'évaluation) n'incluent pas les pénalités/bonus afin de mesurer le vrai score ALE.

---

## Réseau de neurones — DQN / Dueling DQN

Le backbone CNN est commun aux deux variantes :

| Couche | Paramètres | Sortie |
|--------|-----------|--------|
| Conv2d | 32 filtres 8×8, stride 4 | 20×20 |
| Conv2d | 64 filtres 4×4, stride 2 | 9×9 |
| Conv2d | 64 filtres 3×3, stride 1 | 7×7 |
| Flatten | — | 3 136 |

**Mode classique (`USE_DUELING_DQN=False`)** : `fc(3136→512→9)` → Q(s,a)

**Mode Dueling (`USE_DUELING_DQN=True`)** :
- `value_stream(3136→512→1)` → V(s)
- `advantage_stream(3136→512→9)` → A(s,a)
- Combinaison : **Q(s,a) = V(s) + A(s,a) − mean(A)**

La soustraction de la moyenne centre les avantages et stabilise l'entraînement dans les corridors où toutes les actions sont équivalentes.

---

## Hyperparamètres

| Paramètre | Valeur |
|-----------|--------|
| Buffer size | 100 000 transitions |
| Batch size | 32 |
| Gamma (discount) | 0.99 |
| Learning rate | 2.5 × 10⁻⁵ |
| Learning starts | 20 000 steps |
| Train frequency | toutes les 4 actions |
| Target sync | toutes les 2 000 steps |
| ε-start → ε-end | 1.0 → 0.05 sur 1 500 épisodes |
| Max steps/épisode | 5 000 |

---

## Flags d'améliorations (train.py)

Les quatre améliorations sont activables indépendamment :

```python
USE_DOUBLE_DQN    = True   # corrige la sur-estimation des Q-values
USE_DUELING_DQN   = True   # sépare V(s) et A(s,a)
USE_PER           = True   # sampling priorisé par |TD-error|
USE_TROPHY_BUFFER = True   # boost rétroactif des épisodes exceptionnels  
```

### Double DQN

**Problème DQN classique** : `target_net` sélectionne ET évalue la meilleure action suivante via `max Q_target(s', a')`, ce qui crée un biais positif systématique (Q-values gonflées).

**Solution** : découplage des deux rôles :
1. `policy_net` choisit l'action : `a* = argmax Q_online(s', a)`
2. `target_net` évalue cette action : `Q_target(s', a*)`

### Prioritized Experience Replay (PER)

**Problème** : le sampling uniforme traite toutes les transitions également, alors que 95 % d'entre elles sont "manger une gomme" (TD ≈ 0) et 5 % sont cruciales (power pellet, mort, fantôme mangé).

**Solution** : chaque transition reçoit une priorité `p_i = (|TD_i| + ε)^α`. Le sampling est proportionnel à cette priorité. Les IS-weights corrigent le biais introduit :

```
w_i = (N · P(i))^(−β)
loss = Σ w_i · Huber(TD_i)
```

Paramètres : `α=0.6`, `β_start=0.4`, `ε=1×10⁻⁶`.

---

## Contributions originales

###   Système de récompenses configurable (`reward.json`)

Les récompenses brutes ALE sont remplacées par des valeurs configurables dans `reward.json`, rechargé à chaque épisode (hot-reload). Cela permet d'ajuster le comportement de l'agent sans relancer l'entraînement.

```json
{
    "level_clear_bonus": 50.0,
    "dots_level": 158,
    "pastille_normale": 10,
    "pastille_grosse": 50
}
```

Le wrapper `MaxAndSkipEnv` intercepte les récompenses ALE et les remplace :
- `r = 10` (gomme normale) → `pastille_normale` pts
- `r = 50` (power pellet) → `pastille_grosse` pts
- `r ∈ {200, 400, 800, 1600}` (fantôme) → récompense configurable
- Level clear (158 gommes mangées) → bonus `+50` pts via `LevelClearBonusEnv`

---

###   Trophy Buffer

**Problème PER seul** : les priorités sont purement locales (TD-error à l'instant t). Une transition dans un épisode exceptionnel qui précède un enchaînement brillant peut avoir une TD-error faible → elle sera rarement rejouée malgré sa valeur réelle pour l'apprentissage.

**Solution** : en fin d'épisode, un **boost rétroactif** est appliqué à toutes les transitions de cet épisode si son return total `G_t` dépasse la baseline :

```
boost = 1 + λ · sigmoid((G_t − baseline) / scale)
```

La baseline est une **EMA (Exponential Moving Average)** des returns observés :

```
baseline ← (1 − α) · baseline + α · G_t     avec α = 0.01
```

Résultat : les épisodes "trophée" (G_t >> baseline) voient leurs transitions propulsées dans les top priorités du buffer PER, et sont rejouées bien plus souvent lors des prochains mini-batchs.

**Paramètres Trophy Buffer** :

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `TROPHY_LAMBDA` | 0.5 | Intensité du boost ∈ [1.0, 1.5] |
| `TROPHY_BASELINE_ALPHA` | 0.01 | Réactivité de la baseline EMA |
| `TROPHY_MIN_DELTA` | 5 | Seuil minimum au-dessus de la baseline |

**Nécessite `USE_PER=True`** — le Trophy Buffer est une extension du PER qui tague chaque transition avec un `episode_id` au moment du `push()`.

---

## Résultats

L'agent a enregistré plusieurs vidéos de parties où il dépasse 120 gommes mangées, dont le meilleur épisode atteignant **180 gommes** (niveau quasiment complété). Les enregistrements sont disponibles dans `videos/`.

| Fichier | Dots | Épisode |
|---------|------|---------|
| pacman_win_180dots_ep27.mp4 | 180 | 27 |
| pacman_win_176dots_ep18.mp4 | 176 | 18 |
| pacman_win_143dots_ep8.mp4 | 143 | 8 |
| pacman_win_131dots_ep7.mp4 | 131 | 7 |

---

## Utilisation

```bash
make help
```

 ENTRAINEMENT
    make train          Lance train.py (reprend si checkpoint)

  TEST ET VISUALISATION
    make test           Partie visuelle epsilon=0
    make plot           Courbes depuis log.json
    make record         Enregistre parties >= 159 dots en MP4

  DOCUMENTATION
    make doc            Genere HTML Doxygen dans docs/
    make doc-open       Genere + ouvre dans le navigateur

  UTILITAIRES
    make status         Stats entrainement (best, loss, eps...)
    make backup         Copie checkpoint avec timestamp
    make list-backups   Liste les .pth disponibles
    make check-env      Verifie Python, CUDA, dependances
    make install        pip install toutes les dependances

  NETTOYAGE
    make clean          Supprime checkpoint + log.json
    make clean-videos   Supprime les MP4 enregistres
    make clean-doc      Supprime docs/ et Doxyfile
    make clean-all      Supprime TOUT (avec confirmation)

  Workflow typique :
    make install        # Installer les dependances
    make check-env      # Verifier CUDA / Python
    make train          # Entrainer l'agent
    make status         # Verifier la progression
    make plot           # Visualiser les courbes
    make test           # Voir l'agent jouer
    make backup         # Sauvegarder le checkpoint
    make doc            # Generer la documentation


# manuelement
```bash
# Entraînement
python train.py

# Évaluation
python test_dqn.py

# Enregistrement vidéo du meilleur agent
python record_best.py

# Visualisation des courbes
python plot_progress.py
```

---

## Dépendances

```
torch >= 2.0
torchvision
gymnasium
ale-py
Pillow
numpy
matplotlib
imageio
```
install torch torchvision numpy gymnasium ale-py matplotlib imageio Pillow
---

## Références

- Mnih et al. (2015) — *Human-level control through deep reinforcement learning* (DQN)
- Van Hasselt et al. (2016) — *Deep Reinforcement Learning with Double Q-learning*
- Wang et al. (2016) — *Dueling Network Architectures for Deep Reinforcement Learning*
- Schaul et al. (2016) — *Prioritized Experience Replay*