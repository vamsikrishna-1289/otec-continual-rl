# experiments/train_ppo_sequential.py

import os
import sys

# ==========================
# FIX IMPORT PATH (FOR ALL SYSTEMS)
# ==========================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# ==========================
# NOW IMPORTS WILL WORK
# ==========================
from data_processing.sst_data_loader import SSTDataLoader
from environment.otec_env import OTECEnvironment

from stable_baselines3 import PPO


# ==========================
# PATH CONFIGURATION
# ==========================
BASE_PATH = r"C:\Users\DELL\PycharmProjects\CAPSTONE\otec-continual-rl\data"

REGIME_PATHS = {
    "T1": os.path.join(BASE_PATH, "T1"),
    "T2": os.path.join(BASE_PATH, "T2"),
    "T3": os.path.join(BASE_PATH, "T3"),
    "T4": os.path.join(BASE_PATH, "T4"),
}


# ==========================
# LOAD DATA
# ==========================
def load_all_regimes():
    regime_data = {}

    for key, path in REGIME_PATHS.items():
        print(f"\nLoading {key}...")

        loader = SSTDataLoader(folder_path=path)
        data = loader.load_and_process()

        regime_data[key] = data

    return regime_data


# ==========================
# TRAIN SEQUENTIALLY
# ==========================
def train_sequential_ppo(regime_data, total_timesteps=3000):
    model = None

    for regime in ["T1", "T2", "T3", "T4"]:
        print(f"\n========== Training on {regime} ==========")

        env = OTECEnvironment(regime_data[regime], regime_type=regime)

        if model is None:
            model = PPO("MlpPolicy", env, verbose=1)
        else:
            model.set_env(env)

        model.learn(total_timesteps=total_timesteps)

    return model


# ==========================
# EVALUATION
# ==========================
def evaluate_model(model, regime_data):
    results = {}

    for regime in ["T1", "T2", "T3", "T4"]:
        env = OTECEnvironment(regime_data[regime], regime_type=regime)

        state = env.reset()
        total_reward = 0

        for _ in range(len(regime_data[regime])):
            action, _ = model.predict(state)
            state, reward, done, _ = env.step(action)

            total_reward += reward

            if done:
                break

        avg_reward = total_reward / len(regime_data[regime])
        results[regime] = avg_reward

        print(f"{regime} -> Avg Reward: {avg_reward:.4f}")

    return results


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    regime_data = load_all_regimes()

    model = train_sequential_ppo(regime_data)

    print("\n========== FINAL EVALUATION ==========")
    results = evaluate_model(model, regime_data)

    import json

    with open("ppo_final_results.json", "w") as f:
        json.dump(results, f, indent=4)