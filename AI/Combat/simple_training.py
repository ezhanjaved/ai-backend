from gymnasium.envs.registration import register
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np

register(
    id='SimpleCombatGame-v1',
    entry_point='my_env.simple_combat:SimpleCombatEnv',
)

all_envs = gym.envs.registry
for env_id in sorted(all_envs.keys()):
    print(env_id)

env = gym.make('SimpleCombatGame-v1')

model = PPO("MlpPolicy", env, verbose=1, 
            batch_size=64, 
            learning_rate=3e-4,
            ent_coef=0.01,
            policy_kwargs={"net_arch": [128, 128]},
            tensorboard_log="./ppo_combat_tensorboard/")

eval_callback = EvalCallback(env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=10000,
                             deterministic=True, render=False)

model.learn(total_timesteps=500000, callback=eval_callback)

model.save("ppo_simple_combat_game_model")

model = PPO.load("ppo_simple_combat_game_model.zip", env=env)

custom_states = [
    [100, 20, 50, 50, 20, 0, 0],
    [20, 100, 50, 50, 20, 1, 0]
]

for state in custom_states:
    state = np.array(state, dtype=np.float32)
    action, _states = model.predict(state, deterministic=True)
    print(f"For state {state}, predicted action: {action}")