import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import os
import time

# --- PARÁMETROS DE ENTRENAMIENTO ---
TIMESTEPS = 200_000
EVAL_EPISODES = 10
RENDER_SPEED = 0.01  # en segundos

# --- CREAR ENTORNO Y ENVOLTORIOS ---
env = gym.make("LunarLander-v3")
env = DummyVecEnv([lambda: env])
env = VecMonitor(env)

# --- CREAR MODELO ---
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./logs/",
    device="cuda"
)

print("Usando: ", model.device)

# --- ENTRENAR ---
print(f"\nEntrenando por {TIMESTEPS} pasos...\n")
model.learn(total_timesteps=TIMESTEPS)

# --- GUARDAR MODELO ---
os.makedirs("models", exist_ok=True)
model.save("models/dqn_lunarlander")

# --- EVALUAR ---
print(f"\nEvaluando por {EVAL_EPISODES} episodios...\n")
eval_env = gym.make("LunarLander-v3")  # sin render
rewards = []

for ep in range(EVAL_EPISODES):
    obs, _ = eval_env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward
        #time.sleep(RENDER_SPEED)
    print(f"Episodio {ep+1}: Recompensa = {total_reward:.2f}")
    rewards.append(total_reward)

eval_env.close()

# --- GRAFICAR ---
os.makedirs("outputs", exist_ok=True)
plt.plot(rewards)
plt.title("Recompensa total por episodio (Evaluación)")
plt.xlabel("Episodio")
plt.ylabel("Reward")
plt.grid(True)
plt.savefig("outputs/reward_eval.png")
plt.show()