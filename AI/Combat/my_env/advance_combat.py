import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AdvanceCombatEnv(gym.Env):
    def __init__(self):
        super(AdvanceCombatEnv, self).__init__()
        
        self.action_space = spaces.Discrete(6)
        
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, -800, 0, 0, 0]), 
                                            high=np.array([100, 100, 700, 800, 120, 1, 1]), 
                                            dtype=np.float32)        
        self.max_hp = 100         
        self.max_time = 120        
        self.attack_distance = 70   
        self.reward_per_attack = 20
        self.penalty_per_hit = -5
        self.time_penalty = 1    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.ai_hp = self.max_hp
        self.player_hp = self.max_hp
        self.total_distance = np.random.randint(200, 700) 
        self.remaining_time = self.max_time
        self.x_distance = self.total_distance

        self.isAttacking = 0  
        self.knightDefending = 0 

        state = [self.ai_hp, self.player_hp, self.total_distance, self.x_distance, self.remaining_time, self.isAttacking, self.knightDefending]
        obs = np.array(state, dtype=np.float32)
        
        return obs, {}

    def step(self, action):
        reward = 0
        done = False
        truncated = False
        
        if action == 0 or action == 1 or action == 2:  # Player attacking
            self.isAttacking = 1
            self.knightDefending = 0
        elif action == 3:  # Player defending
            self.isAttacking = 0
            self.knightDefending = 1
        else:  # Player is moving 
            self.isAttacking = 0
            self.knightDefending = 0

        # AI's behavior when Player is too far away at this stage we want AI to move towards player
        if self.total_distance > self.attack_distance:
            if self.x_distance > 0:
                if action == 4:
                    self.x_distance = max(-800, self.x_distance - 20)
                    self.total_distance -= 20
            else:   
                if action == 5:
                    self.x_distance = min(800, self.x_distance + 20)
                    self.total_distance -= 20

        # AI's behavior when Player is attacking
        if self.isAttacking == 1 and self.total_distance < self.attack_distance:
            if action == 4 or action == 5:  
                reward += 2
            else:
                reward -= 10
                self.ai_hp -=15

        # AI's behavior when Player is not attacking
        elif self.isAttacking == 0:
            if self.knightDefending == 0 and self.total_distance < self.attack_distance:  
                if action == 0:  # Attack1
                    self.player_hp -= 10
                    reward += self.reward_per_attack * 0.25
                elif action == 1:  # Attack2
                    self.player_hp -= 15
                    reward += self.reward_per_attack * 0.50
                elif action == 2:  # Attack3
                    self.player_hp -= 20
                    reward += self.reward_per_attack * 0.75
                elif action == 3:  # Attack4 (Jump Attack)
                    self.player_hp -= 25
                    reward += self.reward_per_attack
            elif (self.x_distance > 0):
                if action == 4:  # MoveLeft
                    self.x_distance = max(-800, self.x_distance - 20)
                    self.total_distance -= 20
            else: 
                if action == 5:  # MoveRight
                    self.x_distance = min(800, self.x_distance + 20)
                    self.total_distance -= 20

        #AI's behavior when Player is attacking but time is running out for him (this way AI can waste time)
        if self.remaining_time < 30 and self.isAttacking == 1:
            if (self.x_distance > 0):
                if action == 5:  # MoveRight To Evade
                    self.x_distance = min(800, self.x_distance + 20)
                    self.total_distance -= 20
            else:
                if action == 4:  # MoveLeft To Evade
                    self.x_distance = max(-800, self.x_distance - 20)
                    self.total_distance -= 20

        #AI's behavior when Player is attacking and AI's health is little too low
        if self.isAttacking == 1 and self.ai_hp < 20:
            if (self.x_distance > 0):  
                if action == 4:  # MoveLeft
                    self.x_distance = max(-800, self.x_distance - 20)
                    self.total_distance -= 20   
            else:
                if action == 5:  # MoveRight
                    self.x_distance = min(800, self.x_distance + 20)
                    self.total_distance -= 20 

        if self.knightDefending == 1 and self.remaining_time % 5 == 0:
            reward -= 2  # Small penalty for excessive defending

        #if self.total_distance > self.attack_distance and action in [4, 5]:
            #reward += 2  # Reward for moving closer to attack range
            
        reward += self.time_penalty
        self.remaining_time -= 1
        
        if self.player_hp <= 0:
            reward += 50  
            done = True
        elif self.ai_hp <= 0:
            reward -= 50  
            done = True
        elif self.remaining_time <= 0:
            truncated = True
            reward += 25 

        state = [self.ai_hp, self.player_hp, self.total_distance, self.x_distance, self.remaining_time, self.isAttacking, self.knightDefending]
        return np.array(state, dtype=np.float32), reward, done, truncated, {}