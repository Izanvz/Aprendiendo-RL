import gymnasium as gym
from dqn_agent import DQNAgent
from utils import plot_rewards
import torch
import os

# --- Configuración ---
EPISODES = 500
MAX_STEPS = 500
TARGET_UPDATE = 10  # cada cuántos episodios sincronizar redes
ENV_NAME = "CartPole-v1"
SAVE_PATH = "outputs/model_final.pth"

# --- Crear entorno y agente ---
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim)

rewards = []

for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    for t in range(MAX_STEPS):
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

    # Actualizar red objetivo periódicamente
    if episode % TARGET_UPDATE == 0:
        agent.update_target()

    print(f"Episodio {episode+1}: recompensa = {total_reward:.2f}")

# Guardar modelo entrenado
os.makedirs("outputs", exist_ok=True)
torch.save(agent.q_net.state_dict(), SAVE_PATH)

# Graficar resultados
plot_rewards(rewards)
