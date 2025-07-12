import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from datetime import datetime
import os

# --- CONFIGURACIÓN ---
TIMESTEPS = 1_000_000
MODEL_VERSION = f"v1_{TIMESTEPS}"
MODEL_DIR = "models"
LOG_DIR = "logs"
SAVE_PATH = os.path.join(MODEL_DIR, f"dqn_lunarlander_{MODEL_VERSION}")

# --- CREAR ENTORNO MONITORIZADO ---
env = gym.make("LunarLander-v3")
env = DummyVecEnv([lambda: env])
env = VecMonitor(env)

# --- AJUSTES DE RED NEURONAL ---
policy_kwargs = dict(net_arch=[256, 256])

# --- CREAR MODELO ---
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=5e-4,
    buffer_size=100_000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    exploration_fraction=0.3,
    exploration_final_eps=0.02,
    target_update_interval=1000,
    train_freq=1,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=LOG_DIR,
    device="cuda"
)

print(f"\nEntrenando modelo: {SAVE_PATH}")
model.learn(total_timesteps=TIMESTEPS)

# --- GUARDAR MODELO ---
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(SAVE_PATH)
print(f"\n✅ Modelo guardado en: {SAVE_PATH}.zip")
