import gymnasium as gym
from dqn_agent import DQNAgent
from utils import plot_rewards
import torch
import os
print(DQNAgent.__init__.__code__.co_varnames)


# Configuración
ENV = "LunarLander-v2"       # usaremos v2 porque v3 no está disponible
EPISODES = 500
MAX_STEPS = 1000
TARGET_UPDATE = 20
SAVE_PATH = "outputs/lunarlander_model_v2.pth"

# Parámetros epsilon controlados
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY_EPISODES = 300  # bajará a epsilon_min en 300 episodios

# Crear entorno y agente
env = gym.make(ENV)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(
    state_dim, action_dim,
    gamma=0.99,
    lr=1e-3,
    batch_size=64,
    buffer_size=50_000,
    epsilon_start=EPSILON_START,
    epsilon_min=EPSILON_MIN,
    epsilon_decay_linear=True,  # NUEVO: modo lineal activado
    epsilon_decay_episodes=EPSILON_DECAY_EPISODES
)

rewards = []

for ep in range(1, EPISODES + 1):
    state, _ = env.reset()
    total_reward = 0
    for _ in range(MAX_STEPS):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store_transition(state, action, reward, next_state, done)
        agent.train_step()
        state = next_state
        total_reward += reward
        if done:
            break

    rewards.append(total_reward)
    if ep % TARGET_UPDATE == 0:
        agent.update_target()

    print(f"Episodio {ep}/{EPISODES}: Reward = {total_reward:.2f} | ε = {agent.epsilon:.3f}")

# Guardar modelo
os.makedirs("outputs", exist_ok=True)
torch.save(agent.q_net.state_dict(), SAVE_PATH)

# Graficar
plot_rewards(rewards, save_path="outputs/rewards_lunarlander_v2.png")
