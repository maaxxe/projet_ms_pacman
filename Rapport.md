# Rapport de contributions — DQN MsPacman

---

## 1. Description explicite des contributions

### Point de départ : DQN de référence

L'implémentation de base suit fidèlement l'article fondateur de Mnih et al. (2015) :
un réseau CNN à trois couches convolutives suivi d'une tête fully-connected, entraîné
par Q-learning avec un replay buffer uniforme et un target network figé. C'est la
**baseline** contre laquelle toutes les améliorations ont été comparées.

---

### Contribution 1 — Intégration combinée Double DQN + Dueling DQN + PER

Ces trois techniques existent séparément dans la littérature (Rainbow, 2017 les combine
toutes). La contribution ici est leur **intégration modulaire et activable indépendamment**
via des flags booléens dans `train.py`, ce qui permet d'isoler l'effet de chaque
amélioration lors des expérimentations.

**Ce qui a été modifié dans le code par rapport au DQN classique :**

| Fichier | Modification |
|---------|-------------|
| `dqn_model.py` | Ajout du flag `dueling=` dans `DQN.__init__()` ; deux streams séparés (`value_stream`, `advantage_stream`) ; `forward()` unifié |
| `train.py` | Branchement conditionnel `if USE_DOUBLE_DQN` pour découpler sélection et évaluation de l'action cible |
| `replay_buffer.py` | Ajout de `PrioritizedReplayBuffer` complet en parallèle du `ReplayBuffer` uniforme existant |
| `train.py` | IS-weights dans la loss, `update_priorities()` après chaque gradient step |

---

### Contribution 2   — Système de récompenses configurable (reward.json)

**Ce qui existait :** ALE fournit des récompenses brutes basées sur le score du jeu
(10 pts pour une gomme, 50 pour un power pellet, 200–1600 pour un fantôme). La pratique
standard est soit de les utiliser telles quelles, soit de les clipper à {-1, 0, +1}.

**Apport :** Interception des récompenses brutes dans `MaxAndSkipEnv.step()` et
remplacement par des valeurs lues depuis un fichier JSON rechargé à chaque épisode
(*hot-reload*). Cela permet de **reshaper la fonction de récompense sans modifier ni
relancer le code d'entraînement**.

```json
{
   "level_clear_bonus": 50.0,
    "dots_level": 158,
    "pastille_normale": 1.0,
    "pastille_grosse": 3.5,
    "ghost": 0.0
}
```

Un second wrapper, `LevelClearBonusEnv`, compte les gommes mangées via `info["dot_count"]`
et injecte un bonus de +50 pts lorsque les 158 gommes d'un niveau sont complétées. Ce
bonus est placé **après** `MaxAndSkipEnv` et **avant** tout clipping éventuel, afin qu'il
ne soit pas écrasé.
Les fantomes ne rapportent rien pour pas faire de notre fantomes un tueur

---

### Contribution 3   — Trophy Buffer

**Ce qui existait :** PER priorise les transitions selon leur TD-error locale à l'instant t.
Une transition dans un épisode brillant mais dont la TD-error est faible (car l'agent
avait déjà bien prédit cette valeur) sera rarement rejouée malgré sa valeur réelle.

**Apport :** Extension rétroactive du PER. En fin d'épisode, si le return total `G_t`
dépasse la baseline d'un seuil minimum, toutes les transitions de cet épisode voient
leur priorité multipliée par un facteur calculé via une sigmoïde :

```
boost = 1 + λ · sigmoid((G_t − baseline) / scale)
```

La baseline est une moyenne mobile exponentielle (EMA) mise à jour à chaque épisode :

```
baseline ← (1 − 0.01) · baseline + 0.01 · G_t
```

**Ce qui a été implémenté concrètement :**
- Chaque transition reçoit un `episode_id` au `push()` — stocké dans un tableau
  `episode_ids` circulaire dans `PrioritizedReplayBuffer`.
- `boost_episode_priorities(episode_id, G_t, baseline)` retrouve tous les indices
  correspondants via `np.where()` et applique le multiplicateur in-place sur
  `self.priorities`.
- La baseline est initialisée au premier return observé (pas à zéro) pour éviter un
  boost massif des premiers épisodes.

---

### Contribution 4 — Expérimentation de reward shaping avancé (abandonnée)

Tentative d'un système de récompenses élaboré incluant :
- **Pénalité temporelle** : malus proportionnel au nombre de steps passés sans manger
  de gomme, pour forcer l'agent à rester actif.
- **Bonus d'exploration** : récompense positive pour visiter des zones non explorées
  (coins de la carte), modélisé comme un "aimant" vers les tuiles rarement visitées.
- **Bonus de vitesse** : récompense inversement proportionnelle au temps par gomme.

Ces mécanismes ont été **testés puis abandonnés** — voir section 4.

---

## 2. Conditions et paramètres des expérimentations

### Expérience A — DQN classique (baseline)

| Paramètre | Valeur |
|-----------|--------|
| Architecture | CNN 3 couches → fc(512) → Q(9) |
| Buffer | ReplayBuffer uniforme, capacity=100 000 |
| Batch size | 32 |
| Gamma | 0.99 |
| Learning rate | 2.5 × 10⁻⁵ |
| Learning starts | 20 000 steps |
| Train frequency | 4 steps |
| Target sync | 2 000 steps |
| ε | 1.0 → 0.05 sur 1 500 épisodes |
| Récompenses | Récompenses ALE brutes (non clippées) |
| Flags actifs | Aucun |

**Observations :** L'agent apprend lentement à manger des gommes mais reste bloqué dans
des comportements répétitifs (tourner en rond dans un couloir). Les Q-values divergent
parfois en début d'entraînement.Fini le niveau mais pas rapide.

---

### Expérience B — DQN + Double + Dueling + PER + reward.json

| Paramètre | Valeur |
|-----------|--------|
| Architecture | CNN 3 couches → value_stream(512→1) + advantage_stream(512→9) |
| Buffer | PrioritizedReplayBuffer, capacity=100 000, α=0.6, β=0.4 |
| Batch size | 32 |
| Gamma | 0.99 |
| Learning rate | 2.5 × 10⁻⁵ |
| Learning starts | 20 000 steps |
| Train frequency | 4 steps |
| Target sync | 2 000 steps |
| ε | 1.0 → 0.05 sur 1 500 épisodes |
| Récompenses | reward.json (pastille=10, power=50, level_clear=+50) |
| Flags actifs | USE_DOUBLE_DQN, USE_DUELING_DQN, USE_PER |

**Observations :** Convergence sensiblement plus rapide. L'agent développe des stratégies
de nettoyage de niveau plus cohérentes.

---

### Expérience C — Expérience B + Trophy Buffer

| Paramètre | Valeur (en plus de B) |
|-----------|----------------------|
| TROPHY_LAMBDA | 0.5 |
| TROPHY_BASELINE_ALPHA | 0.01 |
| TROPHY_MIN_DELTA | 5 |
| Flags actifs | tous (B + USE_TROPHY_BUFFER) |

**Observations :** Les épisodes exceptionnels sont mieux mémorisés. La baseline EMA
converge progressivement et le boost devient de moins en moins fréquent à mesure
que l'agent progresse, ce qui est le comportement attendu.

---

### Expérience D — Reward shaping avancé (abandonnée)

| Récompense ajoutée | Formule tentée |
|--------------------|---------------|
| Pénalité d'inactivité | −k par step sans gomme (k variable) |
| Bonus exploration coins | +r si tuile non visitée depuis N steps |
| Bonus vitesse | +1 / (steps_since_last_dot + 1) |

**Paramètres testés :** k ∈ {0.01, 0.05, 0.1}, N ∈ {50, 100, 200}, r ∈ {0.5, 1.0, 2.0}.
Plusieurs combinaisons testées sur 500–1000 épisodes chacune.

**Observations :** Dans tous les cas, l'agent perd son comportement de base
et régresse.Il se perd dans trop d'informations de rewards.


### Expérience E — Pac-man tueur (abandonnée)

| Paramètre | Valeur |
|--------------------|---------------|
| Récompense fantôme | config ghost élevée (multiplicateur) |
| Récompense gomme | inchangée |
| Flags actifs | tous (configuration C) |

**Problème observé :** Un enchaînement de 4 fantômes sous power pellet (200→400→800→1600 pts ALE × multiplicateur) rapportait davantage que le bonus de level clear entier (+50). L'agent a appris à farmer les fantômes en boucle plutôt que de progresser sur les gommes, se bloquant définitivement sur les premiers niveaux.
**Résolution :** ghost remis à 0 dans reward.json ,le level clear redevient l'objectif dominant.

---

## 3. Conclusions des expérimentations

### Ce qui a fonctionné

La combinaison **Double DQN + Dueling + PER + reward.json** apporte une amélioration
nette et stable par rapport à la baseline. Les effets sont complémentaires : Double DQN
stabilise les valeurs cibles, Dueling améliore la généralisation dans les couloirs,
PER accélère l'apprentissage des transitions rares mais cruciales.

Le **Trophy Buffer** contribue à mémoriser les stratégies gagnantes. L'EMA de la
baseline s'adapte progressivement au niveau de l'agent, ce qui rend le mécanisme
robuste sur la durée : les boosts sont fréquents en début d'entraînement (quand
tout bon épisode dépasse la baseline) et deviennent plus sélectifs à mesure que
le niveau général monte.

Le **reward shaping via reward.json** a permis d'orienter le comportement de l'agent
sans réécrire de code. Le bonus level_clear (+50) a notamment encouragé l'agent à
finir les niveaux plutôt que de chasser les fantômes. bonus ghost (+0).

### Ce qui n'a pas fonctionné — Reward shaping avancé

La pénalité d'inactivité, le bonus d'exploration et le bonus de vitesse ont tous
échoué, pour des raisons différentes mais liées :

**Pénalité d'inactivité :** L'agent a appris à éviter la pénalité en se déplaçant
aléatoirement plutôt qu'en cherchant des gommes. La pénalité récompense le mouvement,
pas l'efficacité.

**Bonus d'exploration (aimant vers les coins) :** Le signal d'exploration entre en
compétition directe avec le signal de survie. L'agent a tendance à courir vers les
coins sans stratégie globale, ce qui le mène droit vers les fantômes. La carte de
MsPacman est trop petite pour que l'exploration soit une priorité : les gommes sont
visibles, pas cachées.

**Bonus de vitesse :** Crée un signal bruité et difficile à stabiliser. L'agent
optimise la vitesse locale plutôt que la stratégie globale, ce qui aboutit à des
comportements erratiques.

**Conclusion générale :** Dans un environnement aussi dense que MsPacman (gommes
partout, fantômes omniprésents), ajouter des récompenses auxiliaires bruite le
signal d'apprentissage. La récompense naturelle du jeu (score ALE) est déjà bien
structurée. Le bon ajustement est minimaliste : repondérer les récompenses existantes
(reward.json) plutôt que d'en inventer de nouvelles.

---

## 4. Valeurs ajoutées et faiblesses

### Valeurs ajoutées

**Modularité des améliorations.** Les flags booléens permettent d'activer ou désactiver
chaque technique indépendamment. Cela rend le code lisible, testable et reproductible.
Chaque flag est documenté avec son impact théorique et les lignes de code concernées.

**reward.json hot-reload.** Pouvoir modifier les récompenses sans relancer l'entraînement
est un gain pratique important. La config est rechargée à chaque début d'épisode, ce
qui permet d'ajuster finement le comportement en cours de run.

**Trophy Buffer.** L'idée de prioriser rétroactivement les épisodes exceptionnels est
conceptuellement solide et peu coûteuse en calcul (`np.where()` sur un tableau déjà
en mémoire). Elle comble un angle mort de PER : les transitions "correctement prédites"
dans un bon épisode ont une TD-error faible et seraient normalement oubliées.

**Pénalité vie perdue.** `LifeLostPenaltyEnv` injecte un signal négatif clair à chaque
mort, distinct du reward shaping des gommes. Cela encourage la survie sans brouiller
le signal de collecte.

---

### Faiblesses

**Absence d'ablation study systématique.** Les améliorations ont été testées groupées
plutôt qu'individuellement puis combinées de manière contrôlée. Il est donc difficile
de quantifier précisément l'apport de chacune.

**Trophy Buffer non validé de façon isolée.** Son effet est difficile à mesurer séparément
du PER. Il faudrait comparer PER seul vs PER+Trophy sur un grand nombre d'épisodes avec
les mêmes seeds aléatoires.

**β de PER fixe.** Le paramètre β (IS-weights) devrait théoriquement croître de 0.4 vers
1.0 au cours de l'entraînement pour corriger progressivement le biais de sampling. Ici
il reste constant à 0.4, ce qui sous-corrige le biais en fin d'entraînement.

**Pas de normalisation de la baseline Trophy par l'écart-type.** La sigmoïde utilise une
échelle fixe (`scale=100`). Si la distribution des scores change fortement en cours
d'entraînement, l'échelle devient inadaptée.

**Entraînement limité à un seul environnement.** La généralisation des réglages
(reward.json, TROPHY_LAMBDA, etc.) à d'autres jeux Atari n'a pas été testée.

---

## 5. Critiques constructives — Ce qui aurait pu être fait autrement

### Sur le Trophy Buffer

La baseline EMA avec α=0.01 a une mémoire très longue (~100 épisodes). En début
d'entraînement, quand les scores varient énormément, elle réagit trop lentement.
**Alternative :** utiliser un percentile glissant (ex. médiane sur les 50 derniers
épisodes) plutôt qu'une EMA, ce qui est plus robuste aux outliers et ne nécessite pas
de régler α.

Une amélioration supplémentaire serait de **décroître λ au cours du temps** : les boosts
sont utiles en début d'entraînement pour ancrer les bonnes stratégies, mais peuvent
créer un biais de sur-focalisation sur des épisodes anciens en fin d'entraînement
quand le niveau général a augmenté.

### Sur le reward shaping avancé

L'aimant vers les coins aurait pu être implémenté différemment : plutôt qu'une
récompense intrinsèque ajoutée à chaque step, utiliser un **count-based exploration
bonus** décroissant (type `1/sqrt(n(s))`) qui s'éteint naturellement une fois la zone
visitée suffisamment. Cela évite la compétition permanente avec le signal de survie.

La pénalité d'inactivité aurait été plus pertinente avec un **seuil de tolérance**
(ex. la pénalité ne s'active qu'après 50 steps sans gomme, pas dès le premier step),
ce qui laisse à l'agent le temps de contourner un fantôme sans être pénalisé.

Trop de panalisation et rewards en meme temps, l'agent ne sait plus quoi faire.

### Sur l'architecture d'entraînement

**Évaluation périodique séparée.** Le score loggé est celui de l'épisode d'entraînement
(avec ε-greedy). Il serait plus informatif d'intercaler des épisodes d'évaluation purs
(ε=0) toutes les N épisodes pour mesurer la politique greedy réelle, sans bruit
d'exploration.

**Curriculum learning.** MsPacman commence toujours depuis le niveau 1 dans le même
état. Charger l'agent directement sur des niveaux plus avancés (via ALE save states)
pourrait accélérer l'apprentissage des comportements tardifs (poursuite de fantômes
vulnérables, gestion de plusieurs vies).

**β annealing dans PER.** Implémenter la montée progressive de β de 0.4 → 1.0 sur la
durée de l'entraînement est un ajout simple qui améliorerait la correction du biais
de sampling en fin de run.

### Sur la reproductibilité

Fixer les seeds NumPy, Python et CUDA (`torch.manual_seed`, `np.random.seed`,
`random.seed`) et logger la seed dans le checkpoint permettrait de reproduire
exactement n'importe quel run.



---

## Synthèse

| Contribution | Statut | Impact observé |
|---|---|---|
| DQN classique (baseline) | Implémenté | Référence |
| Double DQN | Implémenté + actif | Stabilisation Q-values |
| Dueling DQN | Implémenté + actif | Meilleure généralisation |
| PER | Implémenté + actif | Convergence plus rapide |
| reward.json hot-reload | Original   | Flexibilité sans relancer |
| Level clear bonus | Original   | Encourage finir les niveaux |
| Trophy Buffer | Original   | Meilleure rétention des épisodes clés |
| Reward shaping avancé | Testé, abandonné | Déstabilise l'agent |