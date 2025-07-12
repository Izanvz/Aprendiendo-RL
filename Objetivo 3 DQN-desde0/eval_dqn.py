import gymnasium as gym
import torch
from dqn_agent import QNetwork
import time

# --- Configuración ---
ENV_NAME = "CartPole-v1"
MODEL_PATH = "outputs/model_final.pth"
EPISODES = 10
DELAY = 0.002  # segundos entre frames (ajusta si va muy rápido o lento)

# --- Crear entorno con visualización ---
env = gym.make(ENV_NAME, render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# --- Cargar red entrenada ---
model = QNetwork(state_dim, action_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# --- Evaluar ---
for ep in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = model(state_tensor).argmax().item()

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        time.sleep(DELAY)

    print(f"Episodio {ep+1}: Recompensa = {total_reward:.2f}")

env.close()
