# experiments/plot_results_from_file.py

import json
import matplotlib.pyplot as plt
import numpy as np

# ==========================
# LOAD REAL RESULTS
# ==========================

with open("ppo_final_results.json", "r") as f:
    ppo = json.load(f)

with open("ewc_final_results.json", "r") as f:
    ewc = json.load(f)

tasks = ["T1", "T2", "T3", "T4"]

ppo_values = np.array([ppo[t] for t in tasks])
ewc_values = np.array([ewc[t] for t in tasks])

x = np.arange(len(tasks))

# ==========================
# PLOT
# ==========================
plt.figure(figsize=(10, 6))

plt.plot(x, ppo_values, marker='o', linestyle='--', linewidth=2, label="PPO")
plt.plot(x, ewc_values, marker='s', linestyle='-', linewidth=2, label="PPO + EWC")

plt.fill_between(x, ppo_values, alpha=0.1)
plt.fill_between(x, ewc_values, alpha=0.1)

plt.xticks(x, tasks)
plt.axhline(0, linestyle=':', linewidth=1)

plt.title("Performance Comparison (Real Experiment Data)")
plt.xlabel("Tasks")
plt.ylabel("Average Reward")

plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig("final_results_real.png", dpi=300)

plt.show()