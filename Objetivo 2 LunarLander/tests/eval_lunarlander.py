import gymnasium as gym
from stable_baselines3 import DQN
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# --- CONFIGURACIÓN ---
MODEL_PATH = "models/dqn_lunarlander_v1_1000000"
EVAL_EPISODES = 1000
RENDER_EPISODES = 2
FAST_EVAL = True  # True = sin render, False = con render

# --- CARGAR MODELO ---
assert os.path.exists(MODEL_PATH + ".zip"), f"Modelo no encontrado en {MODEL_PATH}.zip"
model = DQN.load(MODEL_PATH)
print("Modelo cargado:", MODEL_PATH)

# --- EVALUACIÓN RÁPIDA (sin render) ---
print(f"\nEvaluando rendimiento en {EVAL_EPISODES} episodios...\n")
env = gym.make("LunarLander-v3")  # sin render

rewards = []
for ep in range(EVAL_EPISODES):
    print("Eval: ",ep)
    obs, _ = env.reset()
    done = False
    total = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total += reward
    rewards.append(total)

env.close()

# --- RESULTADOS ESTADÍSTICOS ---
mean_reward = np.mean(rewards)
std_reward = np.std(rewards)
print(f"Recompensa media: {mean_reward:.2f}")
print(f"Desviación estándar: {std_reward:.2f}")
print(f"Máximo: {np.max(rewards):.2f}")
print(f"Mínimo: {np.min(rewards):.2f}")

# --- GRAFICAR ---
os.makedirs("outputs", exist_ok=True)
plt.plot(rewards)
plt.title("Recompensas por episodio (Evaluación sin render)")
plt.xlabel("Episodio")
plt.ylabel("Reward")
plt.grid(True)
plt.savefig("outputs/rewards_stat_eval.png")
plt.show()

# --- VISUALIZACIÓN DE EPISODIOS ---
print(f"\nVisualizando {RENDER_EPISODES} episodios...\n")
vis_env = gym.make("LunarLander-v3", render_mode="human")

for ep in range(RENDER_EPISODES):
    obs, _ = vis_env.reset()
    done = False
    total = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = vis_env.step(action)
        done = terminated or truncated
        total += reward
        time.sleep(0.01)  # control de framerate
    print(f"[Visual] Episodio {ep+1}: Recompensa = {total:.2f}")

vis_env.close()
