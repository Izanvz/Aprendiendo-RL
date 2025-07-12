import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# Crear entorno y envolverlo con monitor
env = gym.make("LunarLander-v3")
env = Monitor(env)
env = DummyVecEnv([lambda: env])
env = VecMonitor(env)

# Entrenar el agente
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
model.learn(total_timesteps=50_000)

# Guardar el modelo
model.save("dqn_lunarlander")

# Cargar y testear
model = DQN.load("dqn_lunarlander")
obs = env.reset()

rewards = []
for _ in range(10):  # ejecutar 10 episodios
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
    rewards.append(total_reward)

env.close()

# Visualizar recompensas por episodio
plt.plot(rewards)
plt.title("Recompensas por episodio (evaluación)")
plt.xlabel("Episodio")
plt.ylabel("Recompensa total")
plt.savefig("rewards_plot.png")
plt.show()
