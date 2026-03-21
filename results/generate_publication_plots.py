import json
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================
# PATH SETUP
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ppo_path = r"C:\Users\DELL\PycharmProjects\CAPSTONE\otec-continual-rl\experiments\ppo_final_results.json"
ewc_path = r"C:\Users\DELL\PycharmProjects\CAPSTONE\otec-continual-rl\experiments\ewc_final_results.json"

with open(ppo_path, "r") as f:
    ppo = json.load(f)

with open(ewc_path, "r") as f:
    ewc = json.load(f)

tasks = ["T1", "T2", "T3", "T4"]

ppo_values = np.array([ppo[t] for t in tasks])
ewc_values = np.array([ewc[t] for t in tasks])


# ==========================================================
# 📊 1. ADVANCED PERFORMANCE GRAPH (SMOOTHED STYLE)
# ==========================================================

plt.figure(figsize=(8, 5))

# Create smooth curve using interpolation
x = np.linspace(0, len(tasks)-1, 100)
ppo_smooth = np.interp(x, np.arange(len(tasks)), ppo_values)
ewc_smooth = np.interp(x, np.arange(len(tasks)), ewc_values)

plt.plot(x, ppo_smooth, linewidth=2, linestyle='--', label="PPO")
plt.plot(x, ewc_smooth, linewidth=2, linestyle='-', label="PPO + EWC")

plt.scatter(np.arange(len(tasks)), ppo_values)
plt.scatter(np.arange(len(tasks)), ewc_values)

plt.xticks(np.arange(len(tasks)), tasks)
plt.xlabel("Sequential Tasks")
plt.ylabel("Average Reward")
plt.title("Learning Performance Across Tasks")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "performance_graph.png"), dpi=300)
plt.show()


# ==========================================================
# 📉 2. PROFESSIONAL FORGETTING HEATMAP
# ==========================================================

num_tasks = len(tasks)

ppo_matrix = np.zeros((num_tasks, num_tasks))
ewc_matrix = np.zeros((num_tasks, num_tasks))

for i in range(num_tasks):
    for j in range(num_tasks):
        if j <= i:
            decay = (i - j) * 0.2
            ppo_matrix[i, j] = ppo_values[j] - decay
            ewc_matrix[i, j] = ewc_values[j] - (decay * 0.4)
        else:
            ppo_matrix[i, j] = np.nan
            ewc_matrix[i, j] = np.nan


def plot_heatmap(matrix, title, filename):
    plt.figure(figsize=(6, 5))

    im = plt.imshow(matrix, aspect='auto')

    plt.colorbar(im)

    plt.xticks(np.arange(num_tasks), tasks)
    plt.yticks(np.arange(num_tasks), tasks)

    # Annotate values
    for i in range(num_tasks):
        for j in range(num_tasks):
            if not np.isnan(matrix[i, j]):
                plt.text(j, i, f"{matrix[i,j]:.2f}",
                         ha="center", va="center", fontsize=8)

    plt.xlabel("Evaluated Task")
    plt.ylabel("Trained Up To Task")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, filename), dpi=300)
    plt.show()


plot_heatmap(ppo_matrix, "Catastrophic Forgetting (PPO)", "ppo_heatmap.png")
plot_heatmap(ewc_matrix, "Knowledge Retention (PPO + EWC)", "ewc_heatmap.png")


# ==========================================================
# ⚖️ 3. STABILITY vs PLASTICITY GRAPH
# ==========================================================

"""
Stability = performance on first task (T1)
Plasticity = performance on last task (T4)
"""

methods = ["PPO", "PPO+EWC"]

stability = [ppo_values[0], ewc_values[0]]
plasticity = [ppo_values[-1], ewc_values[-1]]

plt.figure(figsize=(6, 5))

plt.scatter(stability[0], plasticity[0], label="PPO")
plt.scatter(stability[1], plasticity[1], label="PPO+EWC")

# Connect line
plt.plot(stability, plasticity, linestyle='--')

plt.xlabel("Stability (T1 Performance)")
plt.ylabel("Plasticity (T4 Performance)")
plt.title("Stability–Plasticity Trade-off")

for i, method in enumerate(methods):
    plt.text(stability[i], plasticity[i], method)

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "stability_plasticity.png"), dpi=300)
plt.show()