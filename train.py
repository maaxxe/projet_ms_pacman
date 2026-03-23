"""
@file      train.py
@brief     Boucle d'entraînement DQN pour MsPacman (ALE/MsPacman-v5).
@details   Intègre 3 améliorations activables indépendamment par des flags booléens :
           - USE_DOUBLE_DQN  : corrige la sur-estimation des Q-values.
           - USE_DUELING_DQN : sépare l'estimation de V(s) et A(s,a).
           - USE_PER         : prioritise les transitions par |TD-error|.
           Supporte la reprise automatique depuis checkpoint avec conversion
           de l'ancien format (net/head → conv/fc).
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

# Reproductibilité CUDA
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False

# =============================================================================
# CHEMINS FICHIERS
# =============================================================================
BASE_DIR            = Path(__file__).resolve().parent
LOG_PATH            = BASE_DIR / "log.json"
CKPT_DIR            = BASE_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
CKPT_PATH           = CKPT_DIR / "mspacman_dqn.pth"
SAVE_EVERY_EPISODES = 500
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# HYPERPARAMÈTRES DQN
# =============================================================================
TOTAL_EPISODES    = 3000*100  ##< Nombre max d'épisodes d'entraînement
BUFFER_SIZE       = 100_000   ##< Capacité du replay buffer
BATCH_SIZE        = 32        ##< Taille du mini-batch d'apprentissage
GAMMA             = 0.99      ##< Facteur de discount (horizon temporel long)
LR                = 1e-4      ##< Learning rate Adam
LEARNING_STARTS   = 20_000    ##< Steps avant de commencer l'apprentissage
TRAIN_FREQ        = 4         ##< 1 step d'apprentissage toutes les N actions
TARGET_UPDATE_EVERY = 2_000   ##< Fréquence de synchronisation target_net
EPS_START         = 1.0       ##< Epsilon initial (exploration totale)
EPS_END           = 0.05      ##< Epsilon minimal (5% exploration résiduelle)
EPS_DECAY_EPISODES = 1500     ##< Épisodes pour décroître de EPS_START à EPS_END
MAX_EPISODE_STEPS = 20_000    ##< Limite de steps par épisode
DOTS_LEVEL        = 158       ##< Nombre de gommes pour détecter un niveau fini

# =============================================================================
# FLAGS D'AMÉLIORATIONS (True = activé, False = désactivé)
# =============================================================================

## @brief Active Double DQN.
#  @details PROBLÈME DQN classique : le target_net sélectionne ET évalue la
#           meilleure action suivante via max Q_target(s', a'), ce qui crée un
#           biais positif systématique (Q-values gonflées = mauvaises décisions).
#
#           SOLUTION Double DQN : on découple les deux rôles :
#             1. policy_net choisit l'action : a* = argmax Q_online(s', a)
#             2. target_net évalue cette action : Q_target(s', a*)
#           Résultat : +20-30% scores, entraînement plus stable.
#
#           CODE IMPACTÉ : bloc `with torch.no_grad()` dans train().
#             if USE_DOUBLE_DQN:
#               next_act = policy_net(nexts_t).argmax(1, keepdim=True)
#               next_q   = target_net(nexts_t).gather(1, next_act).squeeze(1)
#             else:
#               next_q   = target_net(nexts_t).max(1)[0]  # DQN classique
USE_DOUBLE_DQN = True

## @brief Active Dueling DQN.
#  @details PROBLÈME DQN classique : le réseau doit apprendre Q(s,a) pour chaque
#           action, même quand l'action importe peu (ex: corridor droit = toutes
#           actions équivalentes).
#
#           SOLUTION Dueling : sépare le réseau en deux streams après le CNN :
#             - value_stream     → V(s)    : "cet état est-il bon ?"
#             - advantage_stream → A(s,a)  : "cette action est-elle meilleure ?"
#           Combinaison : Q(s,a) = V(s) + (A(s,a) - mean_a(A(s,*)))
#           Résultat : meilleure généralisation, +15-20% convergence.
#
#           CODE IMPACTÉ : dqn_model.py → DQN.__init__() et forward().
#             policy_net = DQN(obs_shape, n_actions, dueling=USE_DUELING_DQN)
#             target_net = DQN(obs_shape, n_actions, dueling=USE_DUELING_DQN)
USE_DUELING_DQN = True

## @brief Active Prioritized Experience Replay (PER).
#  @details PROBLÈME buffer uniforme : random.sample() traite toutes les
#           transitions également, alors que 95% sont "manger une gomme" (TD ≈ 0)
#           et 5% sont cruciales (power pellet, fantôme mangé, mort = TD élevé).
#
#           SOLUTION PER : priorité p_i = (|TD-error_i| + PER_EPS)^PER_ALPHA.
#           Sampling proportionnel à p_i → les transitions importantes sont
#           revisitées 10-100x plus souvent.
#           IS weights w_i = (N * P(i))^(-PER_BETA) corrigent le biais.
#           Résultat : x2 vitesse d'apprentissage, +50% score max.
#
#           CODE IMPACTÉ :
#             - Instanciation : PrioritizedReplayBuffer au lieu de ReplayBuffer.
#             - Sampling : retourne aussi weights et idxs.
#             - Après optimizer.step() : replay_buffer.update_priorities(idxs, td_err)
#             - Loss : (w_t * td_err).mean() au lieu de td_err.mean()
USE_PER = True

## @brief Exposant de priorité PER (0 = uniforme, 1 = full greedy).
PER_ALPHA      = 0.6
## @brief Exposant IS weight initial (monte vers 1.0 idéalement).
PER_BETA_START = 0.4
## @brief Epsilon anti-priorité-nulle.
PER_EPS        = 1e-6

running = True

def handle_interrupt(sig, frame):
    """
    @brief  Gestionnaire signal SIGINT (Ctrl+C).
    @details Met `running=False` pour sortir proprement de la boucle et
             sauvegarder le checkpoint avant de quitter.
    """
    global running
    running = False
signal.signal(signal.SIGINT, handle_interrupt)


def episode_epsilon(episode_idx):
    """
    @brief  Calcule epsilon pour l'exploration epsilon-greedy.
    @param  episode_idx  Numéro d'épisode courant.
    @return float epsilon dans [EPS_END, EPS_START], décroissance linéaire.
    """
    if episode_idx >= EPS_DECAY_EPISODES: return EPS_END
    return EPS_START + (episode_idx / EPS_DECAY_EPISODES) * (EPS_END - EPS_START)


def ensure_log_file():
    """@brief Crée log.json vide s'il n'existe pas encore."""
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", encoding="utf-8") as f: json.dump({}, f, indent=2)


def load_log():
    """
    @brief  Charge les logs JSON d'épisodes précédents.
    @return Dict {episode_key: metrics_dict}, {} si erreur.
    """
    ensure_log_file()
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except: return {}


def save_log(log_data):
    """
    @brief  Sauvegarde les logs JSON sur disque.
    @param  log_data  Dict complet des métriques par épisode.
    """
    with open(LOG_PATH, "w", encoding="utf-8") as f: json.dump(log_data, f, indent=2)


def save_checkpoint(policy_net, target_net, optimizer, episode, best_score,
                    total_env_steps, log_data):
    """
    @brief  Sauvegarde un checkpoint complet PyTorch.
    @param  policy_net      Réseau principal (poids appris).
    @param  target_net      Réseau cible (copie stable).
    @param  optimizer       État Adam (momentum, lr).
    @param  episode         Numéro d'épisode courant.
    @param  best_score      Meilleur score historique.
    @param  total_env_steps Total de steps env effectués.
    @param  log_data        Logs JSON à inclure dans le checkpoint.
    @details Sauvegarde aussi `config` avec les flags actifs pour référence.
    """
    torch.save({
        "policy_net": policy_net.state_dict(),
        "target_net": target_net.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "episode": episode, "best_score": best_score,
        "total_env_steps": total_env_steps, "log_data": log_data,
        "config": {"double_dqn": USE_DOUBLE_DQN, "dueling": USE_DUELING_DQN, "per": USE_PER}
    }, CKPT_PATH)


def load_checkpoint_safely(policy_net, target_net, optimizer, checkpoint):
    """
    @brief  Chargement de checkpoint robuste, compatible ancien et nouveau format.
    @param  policy_net   Réseau à charger.
    @param  target_net   Réseau cible à charger.
    @param  optimizer    Optimizer à restaurer.
    @param  checkpoint   Dict chargé depuis torch.load().
    @return Tuple (success: bool, mode: str).
    @details Essai 1 : chargement direct (nouveau format conv/fc/value_stream).
             Essai 2 : mapping automatique ancien format (net.* → conv.*, head.* → fc.*).
             Utile quand on charge un .pth entraîné AVANT l'ajout du Dueling.
    """
    try:
        # Nouveau format (conv/fc ou value_stream/advantage_stream)
        policy_net.load_state_dict(checkpoint["policy_net"])
        target_net.load_state_dict(checkpoint["target_net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        return True, "nouveau"
    except RuntimeError:
        try:
            # Ancien format (net.*/head.*) → conversion vers conv.*/fc.*
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
    """
    @brief  Boucle principale d'entraînement DQN MsPacman.
    @details Flux :
             1. Création env + réseaux (Dueling si USE_DUELING_DQN).
             2. Buffer PER ou uniforme (si USE_PER).
             3. Reprise checkpoint avec conversion auto si nécessaire.
             4. Boucle épisodes : action ε-greedy → step env → push buffer.
             5. Apprentissage toutes les TRAIN_FREQ actions :
                - Sample PER ou uniforme.
                - Double DQN ou classique pour les targets.
                - Loss pondérée par IS weights si USE_PER.
                - update_priorities si USE_PER.
             6. Sync target_net toutes les TARGET_UPDATE_EVERY actions.
             7. Log JSON + checkpoint toutes les SAVE_EVERY_EPISODES.
    """
    log_data = load_log()
    env = make_train_env(render_mode=None, clip_rewards=True)

    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    # Instanciation avec flag USE_DUELING_DQN → passe dueling=True/False à DQN
    print(f" Config: DoubleDQN={USE_DOUBLE_DQN}, Dueling={USE_DUELING_DQN}, PER={USE_PER}")
    policy_net = DQN(obs_shape, n_actions, dueling=USE_DUELING_DQN).to(DEVICE)
    target_net = DQN(obs_shape, n_actions, dueling=USE_DUELING_DQN).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

    # Instanciation buffer selon USE_PER
    if USE_PER:
        replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=PER_ALPHA)
        print(" PER activé")
    else:
        replay_buffer = ReplayBuffer(BUFFER_SIZE)

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

    for episode_idx in range(episode, TOTAL_EPISODES):
        if not running: break

        state, _ = env.reset()
        ep_reward, ep_score, ep_steps = 0.0, 0.0, 0
        dots_manges, ghosts_eaten = 0, 0

        epsilon = episode_epsilon(episode_idx)
        done = False

        while not done and ep_steps < MAX_EPISODE_STEPS:
            # Exploration ε-greedy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE) / 255.0
                    action = policy_net(state_t).argmax(1).item()

            next_state, reward, term, trunc, info = env.step(action)
            done = term or trunc

            raw_reward = float(info.get("raw_reward", reward))
            if raw_reward > 0:  dots_manges  += 1
            if raw_reward >= 200: ghosts_eaten += 1

            replay_buffer.push(state, action, reward, next_state, done)

            state, ep_reward, ep_score, ep_steps = (
                next_state, ep_reward + reward, ep_score + raw_reward, ep_steps + 1
            )
            total_env_steps += 1

        # --- APPRENTISSAGE (TRAIN_FREQ=4 : 1 update toutes les 4 actions) ---
        if len(replay_buffer) >= current_learning_starts and total_env_steps % TRAIN_FREQ == 0:

            # Sample PER (avec weights/idxs) ou uniforme (weights=1, idxs=None)
            if USE_PER:
                states, actions, rewards, nexts, dones, weights, idxs = \
                    replay_buffer.sample(BATCH_SIZE, PER_BETA_START)
            else:
                states, actions, rewards, nexts, dones = replay_buffer.sample(BATCH_SIZE)
                weights, idxs = np.ones(BATCH_SIZE), None

            states_t = torch.FloatTensor(states).to(DEVICE) / 255.0
            acts_t   = torch.LongTensor(actions).to(DEVICE).unsqueeze(1)
            rews_t   = torch.FloatTensor(rewards).to(DEVICE)
            nexts_t  = torch.FloatTensor(nexts).to(DEVICE) / 255.0
            dones_t  = torch.FloatTensor(dones).to(DEVICE)
            w_t      = torch.FloatTensor(weights).to(DEVICE)

            q_curr = policy_net(states_t).gather(1, acts_t).squeeze(1)

            with torch.no_grad():
                # USE_DOUBLE_DQN : policy_net choisit, target_net évalue
                # Classique       : target_net choisit ET évalue (biais +)
                if USE_DOUBLE_DQN:
                    next_act = policy_net(nexts_t).argmax(1, keepdim=True)
                    next_q   = target_net(nexts_t).gather(1, next_act).squeeze(1)
                else:
                    next_q = target_net(nexts_t).max(1)[0]
                target_q = rews_t + GAMMA * next_q * (1 - dones_t)

            # Loss pondérée par IS weights (USE_PER) ou uniforme (sinon w=1)
            td_err = nn.SmoothL1Loss(reduction='none')(q_curr, target_q)
            loss   = (w_t * td_err).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
            optimizer.step()

            # Mise à jour des priorités PER avec les nouvelles TD-errors
            if USE_PER:
                replay_buffer.update_priorities(idxs, td_err.detach().cpu().numpy())

            losses_list.append(loss.item())

        # Synchronisation target_net toutes les TARGET_UPDATE_EVERY actions
        if total_env_steps % TARGET_UPDATE_EVERY == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Métriques et logs épisode
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
            print(f"\033[ LEVEL CLEAR! ({dots_manges}/{DOTS_LEVEL})\033[0m")

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
