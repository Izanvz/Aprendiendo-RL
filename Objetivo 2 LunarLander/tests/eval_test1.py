import gymnasium as gym
from stable_baselines3 import DQN
import time

# Cargar modelo
model = DQN.load("models/dqn_lunarlander")

# Crear entorno visual
env = gym.make("LunarLander-v3", render_mode="human")

for episode in range(10):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        time.sleep(0.01)  # control del framerate (ajústalo o coméntalo para aún más velocidad)
    print(f"Episodio {episode+1}: Recompensa = {total_reward:.2f}")

env.close()
