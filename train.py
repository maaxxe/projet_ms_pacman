"""
@file      train.py
@brief     Boucle d'entraînement DQN pour MsPacman (ALE/MsPacman-v5).
@details   Intègre 4 améliorations activables indépendamment par des flags booléens :
           - USE_DOUBLE_DQN    : corrige la sur-estimation des Q-values.
           - USE_DUELING_DQN   : sépare l'estimation de V(s) et A(s,a).
           - USE_PER           : prioritise les transitions par |TD-error|.
           - USE_TROPHY_BUFFER : booste rétroactivement les priorités PER des
                                 transitions issues des épisodes exceptionnels.

           ### Trophy Buffer
           En fin d'épisode, boost_episode_priorities() est appelé sur le buffer
           PER pour multiplier les priorités de toutes les transitions de cet
           épisode par :
           @code
           boost = 1 + TROPHY_LAMBDA · sigmoid((G_t - baseline) / scale)
           @endcode
           La baseline est une moyenne mobile exponentielle (EMA) des returns.
           Les épisodes bien au-dessus de la baseline voient leurs transitions
           revisitées bien plus souvent lors des prochains mini-batchs.

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
#               next_q   = target_net(nexts_t).max(1)[0]
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

## @brief Active le Trophy Buffer (extension du PER).
#  @details PROBLÈME PER seul : les priorités sont purement locales (TD-error
#           instant t). Une transition dans un épisode exceptionnel qui précède
#           un enchaînement brillant peut avoir une TD-error faible et donc une
#           priorité basse → elle sera rarement rejouée malgré sa valeur réelle.
#
#           SOLUTION Trophy Buffer : en fin d'épisode, boost rétroactif de toutes
#           ses transitions selon le return total G_t de l'épisode vs la baseline :
#           @code
#           boost = 1 + TROPHY_LAMBDA · sigmoid((G_t - baseline) / scale)
#           @endcode
#           Les épisodes trophy (G_t >> baseline) voient leurs transitions
#           propulsées dans les top priorités et rejouées intensément.
#
#           NÉCESSITE USE_PER=True.
#           CODE IMPACTÉ :
#             - push() reçoit episode_id=episode_idx pour le tracking.
#             - Fin d'épisode : replay_buffer.boost_episode_priorities() appelé.
#             - Baseline mise à jour par EMA : TROPHY_BASELINE_ALPHA.
USE_TROPHY_BUFFER = True

## @brief Exposant de priorité PER (0 = uniforme, 1 = full greedy).
PER_ALPHA      = 0.6
## @brief Exposant IS weight initial (monte vers 1.0 idéalement).
PER_BETA_START = 0.4
## @brief Epsilon anti-priorité-nulle.
PER_EPS        = 1e-6

## @brief Intensité du boost Trophy (boost ∈ [1.0, 1 + TROPHY_LAMBDA]).
#  @details 0.5 → les épisodes parfaits ont des priorités 1.5× plus élevées.
#           Augmenter si la baseline converge lentement (~1.0 max recommandé).
TROPHY_LAMBDA         = 0.5

## @brief Coefficient EMA pour la mise à jour de la baseline Trophy.
#  @details baseline ← (1 - α)·baseline + α·G_t.
#           α=0.01 : mémoire longue (~100 épisodes), lissage fort.
#           α=0.05 : réactivité plus rapide si la distribution des scores évolue.
TROPHY_BASELINE_ALPHA = 0.01

## @brief Écart minimum au-dessus de la baseline pour déclencher un boost Trophy.
#  @details Évite de booster des épisodes marginalement supérieurs à la baseline.
#           Valeur recommandée : ~10-15% du score moyen observé.
TROPHY_MIN_DELTA      = 30 

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
    if episode_idx >= EPS_DECAY_EPISODES:
        return EPS_END
    return EPS_START + (episode_idx / EPS_DECAY_EPISODES) * (EPS_END - EPS_START)


def ensure_log_file():
    """@brief Crée log.json vide s'il n'existe pas encore."""
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)


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
    except:
        return {}


def save_log(log_data):
    """
    @brief  Sauvegarde les logs JSON sur disque.
    @param  log_data  Dict complet des métriques par épisode.
    """
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)


def save_checkpoint(policy_net, target_net, optimizer, episode, best_score,
                    total_env_steps, log_data, trophy_baseline=0.0):
    """
    @brief  Sauvegarde un checkpoint complet PyTorch.
    @param  policy_net       Réseau principal (poids appris).
    @param  target_net       Réseau cible (copie stable).
    @param  optimizer        État Adam (momentum, lr).
    @param  episode          Numéro d'épisode courant.
    @param  best_score       Meilleur score historique.
    @param  total_env_steps  Total de steps env effectués.
    @param  log_data         Logs JSON à inclure dans le checkpoint.
    @param  trophy_baseline  Baseline EMA Trophy Buffer à sauvegarder.
    @details Sauvegarde aussi `config` avec les flags actifs pour référence.
    """
    torch.save({
        "policy_net":      policy_net.state_dict(),
        "target_net":      target_net.state_dict(),
        "optimizer":       optimizer.state_dict(),
        "episode":         episode,
        "best_score":      best_score,
        "total_env_steps": total_env_steps,
        "log_data":        log_data,
        "trophy_baseline": trophy_baseline,
        "config": {
            "double_dqn":    USE_DOUBLE_DQN,
            "dueling":       USE_DUELING_DQN,
            "per":           USE_PER,
            "trophy_buffer": USE_TROPHY_BUFFER,
        }
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
        policy_net.load_state_dict(checkpoint["policy_net"])
        target_net.load_state_dict(checkpoint["target_net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        return True, "nouveau"
    except RuntimeError:
        try:
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


def _trophy_active():
    """
    @brief  Indique si le Trophy Buffer est actif (PER + Trophy requis).
    @return True si USE_PER et USE_TROPHY_BUFFER sont tous les deux True.
    """
    return USE_PER and USE_TROPHY_BUFFER


def train():
    """ 
    @brief  Boucle principale d'entraînement DQN MsPacman.
    @details Flux :
             1. Création env + réseaux (Dueling si USE_DUELING_DQN).
             2. Buffer PER ou uniforme (si USE_PER).
             3. Reprise checkpoint avec conversion auto si nécessaire.
             4. Boucle épisodes : action ε-greedy → step env → push buffer.
                - Si Trophy actif : push() reçoit episode_id=episode_idx.
             5. Apprentissage toutes les TRAIN_FREQ actions :
                - Sample PER ou uniforme.
                - Double DQN ou classique pour les targets.
                - Loss pondérée par IS weights si USE_PER.
                - update_priorities si USE_PER.
             6. Sync target_net toutes les TARGET_UPDATE_EVERY actions.
             7. Fin d'épisode (si Trophy actif) :
                - Appel boost_episode_priorities(episode_idx, ep_score, baseline).
                - Mise à jour EMA de la baseline.
                - Affichage du boost si épisode trophy (G_t > baseline).
             8. Log JSON + checkpoint toutes les SAVE_EVERY_EPISODES.
    """
    log_data = load_log()
    env      = make_train_env(render_mode=None, clip_rewards=True)

    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    print(f" Config: DoubleDQN={USE_DOUBLE_DQN} | Dueling={USE_DUELING_DQN} "
          f"| PER={USE_PER} | TrophyBuffer={USE_TROPHY_BUFFER}")

    # Réseaux (Dueling ou classique selon le flag)
    policy_net = DQN(obs_shape, n_actions, dueling=USE_DUELING_DQN).to(DEVICE)
    target_net = DQN(obs_shape, n_actions, dueling=USE_DUELING_DQN).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

    # Instanciation buffer selon USE_PER
    if USE_PER:
        replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=PER_ALPHA)
        print(f"  PER activé | TrophyBuffer={'ON' if _trophy_active() else 'OFF (PER only)'}")
    else:
        replay_buffer = ReplayBuffer(BUFFER_SIZE)
        if USE_TROPHY_BUFFER:
            print("USE_TROPHY_BUFFER=True ignoré : nécessite USE_PER=True")

    episode         = len(log_data)
    total_env_steps = 0
    best_score      = max(
        [v.get("score", -float("inf")) for v in log_data.values()],
        default=-float("inf")
    )

    ## @brief Baseline EMA Trophy Buffer (moyenne mobile des returns).
    #  @details Initialisée à 0, puis mise à jour après chaque épisode.
    #           Rechargée depuis le checkpoint si disponible.
    trophy_baseline = 0.0

    if CKPT_PATH.exists():
        checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
        success, mode = load_checkpoint_safely(policy_net, target_net, optimizer, checkpoint)
        if success:
            total_env_steps  = checkpoint.get("total_env_steps", 0)
            best_score       = max(best_score, checkpoint.get("best_score", -float("inf")))
            episode          = checkpoint.get("episode", episode)
            trophy_baseline  = checkpoint.get("trophy_baseline", 0.0)
            print(f"  Checkpoint {mode} | ép={episode} | steps={total_env_steps:,} "
                  f"| best={best_score:.0f} | trophy_baseline={trophy_baseline:.1f}")
        else:
            print("  Reset complet")

    current_learning_starts = BATCH_SIZE if total_env_steps > 0 else LEARNING_STARTS
    recent_scores = []
    losses_list   = []

    print(f" Début | eps={episode_epsilon(episode+1):.3f} | device={DEVICE}")
    try:
        for episode_idx in range(episode, TOTAL_EPISODES):
            if not running:
                break

            state, _  = env.reset()
            ep_reward  = 0.0
            ep_score   = 0.0
            ep_steps   = 0
            dots_manges    = 0
            ghosts_eaten   = 0

            epsilon = episode_epsilon(episode_idx)
            done    = False

            # ------------------------------------------------------------------
            # Boucle intra-épisode
            # ------------------------------------------------------------------
            while not done and ep_steps < MAX_EPISODE_STEPS:

                # Exploration ε-greedy
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE) / 255.0
                        action  = policy_net(state_t).argmax(1).item()

                next_state, reward, term, trunc, info = env.step(action)
                done = term or trunc

                raw_reward = float(info.get("raw_reward", reward))
                if raw_reward > 0:     dots_manges  += 1
                if raw_reward >= 200:  ghosts_eaten += 1

                # Push avec episode_id si Trophy Buffer actif
                if _trophy_active() :
                    replay_buffer.push(
                        state, action, reward, next_state, done,
                        episode_id=episode_idx
                    )
                else:
                    replay_buffer.push(state, action, reward, next_state, done)

                state      = next_state
                ep_reward += reward
                ep_score  += raw_reward
                ep_steps  += 1
                total_env_steps += 1

                # --- APPRENTISSAGE (toutes les TRAIN_FREQ actions) ---
                if (len(replay_buffer) >= current_learning_starts
                        and total_env_steps % TRAIN_FREQ == 0):

                    if USE_PER:
                        states, actions, rewards, nexts, dones, weights, idxs = \
                            replay_buffer.sample(BATCH_SIZE, PER_BETA_START)
                    else:
                        states, actions, rewards, nexts, dones = \
                            replay_buffer.sample(BATCH_SIZE)
                        weights, idxs = np.ones(BATCH_SIZE, dtype=np.float32), None

                    states_t = torch.FloatTensor(states).to(DEVICE) / 255.0
                    acts_t   = torch.LongTensor(actions).to(DEVICE).unsqueeze(1)
                    rews_t   = torch.FloatTensor(rewards).to(DEVICE)
                    nexts_t  = torch.FloatTensor(nexts).to(DEVICE) / 255.0
                    dones_t  = torch.FloatTensor(dones).to(DEVICE)
                    w_t      = torch.FloatTensor(weights).to(DEVICE)

                    q_curr = policy_net(states_t).gather(1, acts_t).squeeze(1)

                    with torch.no_grad():
                        if USE_DOUBLE_DQN:
                            next_act = policy_net(nexts_t).argmax(1, keepdim=True)
                            next_q   = target_net(nexts_t).gather(1, next_act).squeeze(1)
                        else:
                            next_q = target_net(nexts_t).max(1)[0]
                        target_q = rews_t + GAMMA * next_q * (1 - dones_t)

                    td_err = nn.SmoothL1Loss(reduction='none')(q_curr, target_q)
                    loss   = (w_t * td_err).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                    optimizer.step()

                    if USE_PER:
                        replay_buffer.update_priorities(
                            idxs, td_err.detach().cpu().numpy()
                        )

                    losses_list.append(loss.item())

                # Synchronisation target_net
                if total_env_steps % TARGET_UPDATE_EVERY == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            # ------------------------------------------------------------------
            # Fin d'épisode : Trophy Buffer boost + mise à jour baseline
            # ------------------------------------------------------------------
            if _trophy_active():
                if ep_score > trophy_baseline + TROPHY_MIN_DELTA:
                    result = replay_buffer.boost_episode_priorities(
                        episode_id=episode_idx,
                        total_return=ep_score,
                        baseline=trophy_baseline,
                        lambda_boost=TROPHY_LAMBDA
                    )
                    if result is not None:
                        n_boosted, boost_factor = result
                        print(
                            f"\033[96m[TROPHY] Ep {episode_idx} | "
                            f"score={ep_score:.0f} > baseline={trophy_baseline:.0f} | "
                            f"boost=×{boost_factor:.3f} sur {n_boosted} transitions\033[0m"
                        )

                # Mise à jour EMA de la baseline (toujours, trophy ou pas)
                if trophy_baseline == 0.0:
                    trophy_baseline = ep_score
                else:
                    trophy_baseline = (
                        (1 - TROPHY_BASELINE_ALPHA) * trophy_baseline
                        + TROPHY_BASELINE_ALPHA * ep_score
                    )

            # ------------------------------------------------------------------
            # Métriques et logs épisode
            # ------------------------------------------------------------------
            recent_scores.append(ep_score)
            if len(recent_scores) > 10:
                recent_scores.pop(0)
            avg_score = np.mean(recent_scores)
            mean_loss = np.mean(losses_list[-100:]) if losses_list else 0.0

            log_data[f"ep_{episode_idx}"] = {
                "eps":            epsilon,
                "loss":           mean_loss,
                "reward":         ep_reward,
                "score":          ep_score,
                "dots":           dots_manges,
                "ghosts":         ghosts_eaten,
                "steps":          ep_steps,
                "avg10":          avg_score,
                "buffer":         len(replay_buffer),
                "trophy_baseline": trophy_baseline if _trophy_active() else None,
            }

            save_log(log_data)

            if dots_manges >= DOTS_LEVEL:
                print(f"\033[92m★ LEVEL CLEAR! ({dots_manges}/{DOTS_LEVEL})\033[0m")

            print(
                f"Ep {episode_idx:4d} | score={ep_score:6.1f} | dots={dots_manges:3d} "
                f"| ghosts={ghosts_eaten} | eps={epsilon:.3f} | loss={mean_loss:.4f} "
                f"| buf={len(replay_buffer)}"
                + (f" | base={trophy_baseline:.0f}" if _trophy_active() else "")
            )

            if ep_score > best_score:
                best_score = ep_score
                print(f"\033[93m NEW BEST: {best_score:.0f}\033[0m")

            if episode_idx % SAVE_EVERY_EPISODES == 0:
                save_log(log_data)
                save_checkpoint(
                    policy_net, target_net, optimizer, episode_idx,
                    best_score, total_env_steps, log_data, trophy_baseline
                )
                print(f"Saved ep{episode_idx} | trophy_baseline={trophy_baseline:.1f}")

    finally:
        save_log(log_data)
        save_checkpoint(
            policy_net, target_net, optimizer, episode_idx,
            best_score, total_env_steps, log_data, trophy_baseline
        )
        env.close()
        print(" Training terminé")
        print(f" Saved ep{episode_idx}")


if __name__ == "__main__":
    train()