import gymnasium as gym
from dqn_agent import DQNAgent
import torch

# Configuración
ENV = "LunarLander-v2"  # Usa v3 solo si tu instalación lo soporta
MODEL_PATH = "outputs/lunarlander_model_v2.pth"
EPISODES = 5
MAX_STEPS = 500
RENDER = True

# Crear entorno con render
env = gym.make(ENV, render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Cargar agente
agent = DQNAgent(state_dim, action_dim)
agent.q_net.load_state_dict(torch.load(MODEL_PATH))
agent.q_net.eval()
agent.epsilon = 0.0  # Sin exploración

# Evaluación
for ep in range(1, EPISODES + 1):
    state, _ = env.reset()
    total_reward = 0
    for _ in range(MAX_STEPS):
        action = agent.select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"Episodio {ep}: Recompensa = {total_reward:.2f}")

env.close()
