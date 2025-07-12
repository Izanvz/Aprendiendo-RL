import matplotlib.pyplot as plt
import os

def plot_rewards(rewards, save_path="outputs/rewards_plot.png"):
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, label="Recompensa por episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Reward")
    plt.title("Recompensas acumuladas")
    plt.grid(True)
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()
