import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import os

# Crear entorno vectorizado y con monitor
env = gym.make("LunarLander-v3")
env = DummyVecEnv([lambda: env])
env = VecMonitor(env)

# Crear modelo DQN
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./logs/"
)

# Entrenar
model.learn(total_timesteps=50000)

# Guardar modelo
os.makedirs("models", exist_ok=True)
model.save("models/dqn_lunarlander")

# Evaluar
rewards = []
for episode in range(10):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    rewards.append(total_reward)

# Graficar recompensas
plt.plot(rewards)
plt.title("Recompensa total por episodio (Evaluación)")
plt.xlabel("Episodio")
plt.ylabel("Reward")
plt.grid(True)
plt.savefig("rewards_plot.png")
plt.show()

env.close()
