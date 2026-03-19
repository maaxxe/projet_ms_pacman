import gymnasium as gym

# --- CONFIGURATION DES RÉCOMPENSES ---
# Tu peux ajuster ces valeurs pour modifier le comportement de ton agent.
CUSTOM_REWARDS = {
    "dot": 1.0,           # A mangé un petit point (le jeu donne 10 pts)
    "energy_pill": 2.0,   # A mangé une super-pastille (le jeu donne 50 pts)
    "ghost": 5.0,         # A mangé un fantôme vulnérable (le jeu donne 200, 400, etc.)
    "fruit": 3.0,         # A mangé un fruit (le jeu donne 100, 200, etc.)
    "death": -10.0,       # A perdu une vie (pénalité)
    "step_penalty": -0.01, # Petite pénalité de temps à chaque frame pour l'inciter à avancer
    "level_clear": 200.0
}
DOTS_LEVEL = 158
class CustomRewardEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Enregistre le nombre de vies initiales au début de l'épisode (souvent 3 ou 4)
        self.lives = info.get('lives', 3)
        return obs, info

    def step(self, action):
        obs, raw_reward, terminated, truncated, info = self.env.step(action)
        
        # 1. Pénalité de temps par défaut
        custom_reward = CUSTOM_REWARDS["step_penalty"]
        
        # 2. Conversion du score brut d'Atari vers tes propres récompenses
        if raw_reward == 10:
            custom_reward += CUSTOM_REWARDS["dot"]
        elif raw_reward == 50:
            custom_reward += CUSTOM_REWARDS["energy_pill"]
        elif raw_reward in [200, 400, 800, 1600]:
            custom_reward += CUSTOM_REWARDS["ghost"]
        elif raw_reward > 0: 
            # Les autres scores positifs correspondent aux fruits
            custom_reward += CUSTOM_REWARDS["fruit"]

        # 3. Pénalité si l'agent perd une vie
        current_lives = info.get('lives', self.lives)
        if current_lives < self.lives:
            custom_reward += CUSTOM_REWARDS["death"]
            self.lives = current_lives
        if self.dots_eaten == DOTS_LEVEL:
            custom_reward += CUSTOM_REWARDS["level_clear"]
            print(f"\033[93m[REWARD] Jackpot Level Clear accordé ! (+{CUSTOM_REWARDS['level_clear']})\033[0m")
            # On passe à 155 pour ne pas donner le bonus en boucle au prochain step
            self.dots_eaten += 1 

        return obs, custom_reward, terminated, truncated, info
